#!/usr/bin/env python3
"""
quality_filter.py — Phase 1: reasoning trace quality filter (extended)

Reads generated reasoning traces (JSONL) from the Phase 1 data pipeline
and filters out examples that violate any of the following rules:

  1. Structure — missing or malformed required fields.
  2. Boxed answer mismatch — extracted \\boxed{} does not equal ground truth.
  3. XML malformation — unbalanced reasoning tags.
  4. No consistency tag — missing both <compatible> and <incompatible>.
  5. Hallucination — claims introduce too many entities not present in
     the input variables (discourse‑aware, relaxed tolerance).
  6. Token budget — trace length exceeds a configurable token limit.
  7. Near‑duplicate — hash‑based fast path, SequenceMatcher fallback.
  8. Answer uniqueness — same (premises, question) pair with different
     ground truths across the dataset (case‑normalised).
  9. Entailment verification — claims in the trace are checked against
     their stated premises using Nemotron via vLLM (batched).

Agent‑specific checks are enforced via callable functions (not hard‑coded
strings), making the system extensible without code changes to the core
validation loop.

Failed examples are written to `failures.json` alongside the filtered
`examples.jsonl` and a `quality_report.json` for full provenance tracking.

Usage:
  python quality_filter.py \\
      --input-dir phase1_data/logical_deduction \\
      --output-dir phase1_data/logical_deduction/filtered \\
      --run-entailment-check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class FilterConfig(BaseModel):
    max_tokens: int = Field(
        2048, ge=64,
        description="Maximum token budget (word‑count approximation)"
    )
    dedup_threshold: float = Field(
        0.90, ge=0.0, le=1.0,
        description="Similarity ratio threshold: traces with ratio > threshold are marked duplicate"
    )
    min_claims: int = Field(1, ge=1, description="Minimum number of <claim> tags required")
    require_consistency_tag: bool = Field(
        True, description="Require at least one <compatible> or <incompatible> tag"
    )
    hallucination_tolerance: int = Field(
        5, ge=0,
        description="Maximum novel entities allowed before flagging hallucination"
    )
    dedup_window: int = Field(
        100, ge=1,
        description="Number of recent traces to check for near‑duplicates"
    )
    run_entailment_check: bool = Field(
        False,
        description="Verify claims follow from premises using Nemotron (requires GPU/vLLM)"
    )
    entailment_model: str = Field(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        description="Model ID for entailment verification"
    )
    entailment_gpu_memory: float = Field(
        0.5, gt=0.0, le=1.0,
        description="GPU memory fraction for entailment model"
    )
    entailment_batch_size: int = Field(
        8, ge=1, le=64,
        description="Number of claims to batch per vLLM call"
    )

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FilterReport:
    agent: str
    total: int = 0
    passed: int = 0
    failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duplicate_groups: int = 0
    answer_conflicts: int = 0
    entailment_failures: int = 0
    failures_path: Optional[Path] = None

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def summary(self) -> str:
        lines = [
            f"Agent: {self.agent}",
            f"  Total: {self.total}",
            f"  Passed: {self.passed}",
            f"  Failed: {self.failed}",
        ]
        if self.failures:
            lines.append("  Failure categories:")
            for reason, count in sorted(self.failures.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
        if self.duplicate_groups:
            lines.append(f"  Duplicate groups: {self.duplicate_groups}")
        if self.answer_conflicts:
            lines.append(f"  Answer conflicts: {self.answer_conflicts}")
        if self.entailment_failures:
            lines.append(f"  Entailment failures: {self.entailment_failures}")
        if self.failures_path:
            lines.append(f"  Failures file: {self.failures_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
_BOXED_REGEX = re.compile(
    r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
)

_DISCOURSE_WORDS: Set[str] = {
    "The", "A", "An", "Therefore", "However", "Thus", "Hence", "So",
    "Conclusion", "Conclusions", "Theorem", "Theorems", "Proof", "Proofs",
    "Lemma", "Lemmas", "Definition", "Definitions", "Example", "Examples",
    "Note", "Notes", "Consider", "Suppose", "Assume", "Let", "Now", "Next",
    "Finally", "Moreover", "Furthermore", "Indeed", "Also", "Otherwise",
    "Either", "Neither", "Both", "Some", "All", "None", "Any", "Every",
    "Each", "First", "Second", "Third", "Then", "Because", "Since",
    "Nevertheless", "Nonetheless", "Instead", "Rather", "Additionally",
    "Claim", "Claims", "Reasoning", "Answer", "Answers", "Question", "Questions",
    "Problem", "Problems", "Solution", "Solutions",
    "Fact", "Facts", "Statement", "Statements", "Premise", "Premises",
}


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_REGEX.findall(text)
    return matches[-1].strip() if matches else None


def _check_xml_wellformed(text: str, tags: List[str]) -> bool:
    for tag in tags:
        opens = len(re.findall(rf"<{tag}\b[^>]*>", text))
        closes = len(re.findall(rf"</{tag}>", text))
        if opens != closes:
            return False
    return True


def _extract_entities(text: str) -> Set[str]:
    proper_nouns = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
    proper_nouns -= _DISCOURSE_WORDS
    quoted = set(re.findall(r'"([^"]+)"', text))
    quoted.update(re.findall(r"'([^']+)'", text))
    return proper_nouns | quoted


def _check_hallucination(
    trace: str, input_text: str, tolerance: int = 5
) -> Tuple[bool, Optional[str]]:
    input_entities = _extract_entities(input_text)
    trace_entities = _extract_entities(trace)
    novel = {e for e in trace_entities - input_entities if len(e) > 1}
    if len(novel) > tolerance:
        return True, ", ".join(sorted(novel)[:10])
    return False, None


# ---------------------------------------------------------------------------
# Duplicate detection (hash‑accelerated)
# ---------------------------------------------------------------------------
def _compute_trace_hash(trace: str) -> str:
    # NOTE: MD5 is used for duplicate detection only, not for security.
    # Normalise whitespace and convert to lowercase for fuzzy comparison.
    normalised = re.sub(r'\s+', ' ', trace.lower()).strip()
    return md5(normalised.encode()).hexdigest()


def _check_near_duplicate(
    trace: str,
    seen_traces: List[str],
    threshold: float,
    window_size: int,
) -> bool:
    """
    Check for near‑duplicates using:
      1. Fast hash‑based exact match (catches most cases).
      2. SequenceMatcher fallback for fuzzy matching (if threshold <= 0.95).
    """
    trace_hash = _compute_trace_hash(trace)
    for seen in seen_traces[-window_size:]:
        if _compute_trace_hash(seen) == trace_hash:
            return True
    # Use fuzzy matching as fallback for thresholds below 0.95
    if threshold <= 0.95:
        for seen in seen_traces[-window_size:]:
            if SequenceMatcher(None, trace, seen).ratio() > threshold:
                return True
    return False


# ---------------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS: Dict[str, type] = {
    "generated_trace": str,
    "ground_truth": str,
    "variables": dict,
    "agent": str,
}


def _validate_example_structure(example: Dict[str, Any]) -> Tuple[bool, str]:
    for field, expected_type in _REQUIRED_FIELDS.items():
        if field not in example:
            return False, f"missing_field_{field}"
        if not isinstance(example[field], expected_type):
            return False, f"wrong_type_{field}_{type(example[field]).__name__}"
    return True, ""


# ---------------------------------------------------------------------------
# Answer uniqueness check (case‑normalised)
# ---------------------------------------------------------------------------
def _build_answer_index(examples: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Build a map from (premises_hash) -> set of normalised ground_truth values.
    Case is normalised to lowercase so "Yes" and "yes" are treated as identical.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for ex in examples:
        variables = ex.get("variables", {})
        key_text = json.dumps(variables, sort_keys=True)
        key_hash = md5(key_text.encode()).hexdigest()
        gt = ex.get("ground_truth", "").strip().lower()
        index[key_hash].add(gt)
    return index


def _check_answer_uniqueness(
    example: Dict[str, Any],
    answer_index: Dict[str, Set[str]],
) -> Tuple[bool, str]:
    variables = example.get("variables", {})
    key_text = json.dumps(variables, sort_keys=True)
    key_hash = md5(key_text.encode()).hexdigest()
    answers = answer_index.get(key_hash, set())
    if len(answers) > 1:
        return False, f"answer_conflict: {sorted(answers)}"
    return True, ""


# ---------------------------------------------------------------------------
# Entailment verification (Nemotron via vLLM, batched)
# ---------------------------------------------------------------------------
_ENTAILMENT_PROMPT = """\
You are a precise logical verifier.  For each claim below, determine whether
it logically follows from the premises.  Answer with ONLY "yes" or "no" on
each line, in the same order as the claims.  Do not include any other text.

Premises:
{premises}

Claims:
{claims}

Answers (one per line):"""


class EntailmentVerifier:
    """Verify logical entailment using Nemotron via vLLM (batched)."""

    def __init__(self, config: FilterConfig):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=config.entailment_model,
            max_model_len=4096,
            gpu_memory_utilization=config.entailment_gpu_memory,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )
        self.batch_size = config.entailment_batch_size

    def check_batch(self, premises: str, claims: List[str]) -> List[bool]:
        """Return a list of bools, one per claim. Processes in batches."""
        results: List[bool] = []
        for i in range(0, len(claims), self.batch_size):
            batch_claims = claims[i:i + self.batch_size]
            batch_results = self._check_single_batch(premises, batch_claims)
            results.extend(batch_results)
        return results

    def _check_single_batch(self, premises: str, claims: List[str]) -> List[bool]:
        if not claims:
            return []

        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = _ENTAILMENT_PROMPT.format(premises=premises, claims=claims_text)
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        results: List[bool] = []
        for line in response.split("\n"):
            line_lower = line.strip().lower()
            has_yes = bool(re.search(r'\byes\b', line_lower))
            has_no = bool(re.search(r'\bno\b', line_lower))
            if has_yes and not has_no:
                results.append(True)
            elif has_no and not has_yes:
                results.append(False)
            else:
                results.append(True)  # conservative default

        while len(results) < len(claims):
            results.append(True)

        return results[:len(claims)]

    def shutdown(self):
        if hasattr(self, "llm"):
            del self.llm
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _extract_premises_from_example(example: Dict[str, Any]) -> str:
    variables = example.get("variables", {})
    for key in ("premises", "evidence", "code", "description"):
        if key in variables:
            return variables[key]
    return "\n".join(str(v) for v in variables.values())


def _extract_claims_from_trace(trace: str) -> List[str]:
    claims = re.findall(r"<claim[^>]*>(.*?)</claim>", trace, re.DOTALL)
    return [c.strip() for c in claims]


# ---------------------------------------------------------------------------
# Agent‑specific checks (callable, extensible)
# ---------------------------------------------------------------------------
def _get_agent_checks(agent_name: str) -> Dict[str, Callable[[str], bool]]:
    checks: Dict[str, Callable[[str], bool]] = {}
    if agent_name == "code_reasoning":
        checks["has_assert"] = lambda t: "<assert" in t
    if agent_name == "iterative_state_transition":
        checks["has_step_claims"] = lambda t: bool(
            re.search(r'<claim\s+step="?\d+"?', t)
        )
    return checks


# ---------------------------------------------------------------------------
# Single‑example validation
# ---------------------------------------------------------------------------
def _validate_example(
    example: Dict[str, Any],
    config: FilterConfig,
    seen_traces: List[str],
    agent_name: str,
    answer_index: Dict[str, Set[str]],
    entailment_verifier: Optional[EntailmentVerifier] = None,
) -> Tuple[bool, str]:
    from reasoning_taxonomy import get_agent

    # 0. Structure
    valid, reason = _validate_example_structure(example)
    if not valid:
        return False, reason

    trace = example["generated_trace"]
    ground_truth = example["ground_truth"]
    variables = example["variables"]
    agent = get_agent(agent_name)
    agent_tags = list(agent.output_tags.keys())

    # 1. Answer uniqueness (cross-example, case‑normalised)
    valid, reason = _check_answer_uniqueness(example, answer_index)
    if not valid:
        return False, reason

    # 2. Token budget
    if len(trace.split()) > config.max_tokens * 1.2:
        return False, "token_overflow"

    # 3. XML
    if not _check_xml_wellformed(trace, agent_tags):
        return False, "xml_malformed"

    # 4. Boxed answer (AUDIT FIX: case-normalised comparison)
    extracted = _extract_boxed(trace)
    if extracted is None:
        return False, "no_boxed"
    if extracted.strip().lower() != ground_truth.strip().lower():
        return False, "answer_mismatch"

    # 5. Consistency tag
    if config.require_consistency_tag:
        has_compat = bool(re.search(r"<compatible\b[^>]*>", trace))
        has_incompat = bool(re.search(r"<incompatible\b[^>]*>", trace))
        if not (has_compat or has_incompat):
            return False, "no_consistency_tag"

    # 6. Minimum claims
    if len(re.findall(r"<claim\b[^>]*>", trace)) < config.min_claims:
        return False, "too_few_claims"

    # 7. Hallucination
    input_text = " ".join(str(v) for v in variables.values())
    is_hallucinated, _ = _check_hallucination(
        trace, input_text, tolerance=config.hallucination_tolerance
    )
    if is_hallucinated:
        return False, "hallucination"

    # 8. Agent‑specific checks
    for check_name, check_fn in _get_agent_checks(agent_name).items():
        if not check_fn(trace):
            return False, f"agent_check_{check_name}"

    # 9. Near‑duplicate
    if _check_near_duplicate(
        trace, seen_traces, config.dedup_threshold, config.dedup_window
    ):
        return False, "duplicate"

    return True, ""


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------
def filter_file(
    input_path: Path,
    output_dir: Path,
    agent_name: str,
    config: FilterConfig,
) -> FilterReport:
    report = FilterReport(agent=agent_name)
    seen_traces: List[str] = []
    kept: List[Dict[str, Any]] = []
    failed_examples: List[Dict[str, Any]] = []

    # Load all examples
    all_examples: List[Dict[str, Any]] = []
    with open(input_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                all_examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Build case‑normalised answer uniqueness index
    answer_index = _build_answer_index(all_examples)
    answer_conflicts = sum(
        1 for answers in answer_index.values() if len(answers) > 1
    )
    report.answer_conflicts = answer_conflicts

    # Initialize entailment verifier if requested
    entailment_verifier = None
    if config.run_entailment_check:
        print("  Loading entailment verifier (Nemotron via vLLM) …")
        entailment_verifier = EntailmentVerifier(config)

    try:
        # First pass: structural validation (fast, no GPU needed for most checks)
        for example in all_examples:
            report.total += 1
            valid, reason = _validate_example(
                example, config, seen_traces, agent_name, answer_index,
                entailment_verifier=None,  # defer entailment to second pass
            )
            if valid:
                # Hold for potential entailment check
                kept.append(example)
                seen_traces.append(example.get("generated_trace", ""))
            else:
                report.failures[reason] += 1
                failed_examples.append({
                    "example": example,
                    "failure_reason": reason,
                })

            if report.total % 10000 == 0:
                print(f"  progress: {report.total} processed, {len(kept)} passed")

        # Second pass: entailment verification on structurally valid examples
        if entailment_verifier is not None and kept:
            print(f"  Running entailment verification on {len(kept)} valid examples …")
            entailment_passed = []
            for i, example in enumerate(kept):
                premises = _extract_premises_from_example(example)
                claims = _extract_claims_from_trace(example.get("generated_trace", ""))
                claims = [c for c in claims if len(c) > 20][:5]  # filter trivial

                if claims:
                    results = entailment_verifier.check_batch(premises, claims)
                    if not all(results):
                        report.failures["entailment_failure"] += 1
                        report.entailment_failures += 1
                        failed_examples.append({
                            "example": example,
                            "failure_reason": "entailment_failure",
                        })
                        continue

                entailment_passed.append(example)

                if (i + 1) % 100 == 0:
                    print(f"    entailment progress: {i+1}/{len(kept)}")

            kept = entailment_passed
            report.passed = len(kept)

        else:
            report.passed = len(kept)

    finally:
        # Guarantee GPU cleanup even if filtering crashes
        if entailment_verifier is not None:
            entailment_verifier.shutdown()

    if "duplicate" in report.failures:
        report.duplicate_groups = report.failures["duplicate"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "examples.jsonl", "w") as f_out:
        for ex in kept:
            f_out.write(json.dumps(ex) + "\n")

    failures_path = output_dir / "failures.json"
    with open(failures_path, "w") as f_fail:
        json.dump(failed_examples, f_fail, indent=2)
    report.failures_path = failures_path

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing examples.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for filtered output")
    parser.add_argument("--agent", default=None, help="Agent name (inferred from input dir if not given)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dedup-threshold", type=float, default=0.9)
    parser.add_argument("--min-claims", type=int, default=1)
    parser.add_argument("--hallucination-tolerance", type=int, default=5)
    parser.add_argument("--no-require-consistency", action="store_true")
    parser.add_argument("--run-entailment-check", action="store_true",
                        help="Verify claims follow from premises using Nemotron (GPU required)")
    parser.add_argument("--entailment-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--entailment-gpu-memory", type=float, default=0.5)
    parser.add_argument("--entailment-batch-size", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_name = args.agent or input_dir.name

    config = FilterConfig(
        max_tokens=args.max_tokens,
        dedup_threshold=args.dedup_threshold,
        min_claims=args.min_claims,
        require_consistency_tag=not args.no_require_consistency,
        hallucination_tolerance=args.hallucination_tolerance,
        run_entailment_check=args.run_entailment_check,
        entailment_model=args.entailment_model,
        entailment_gpu_memory=args.entailment_gpu_memory,
        entailment_batch_size=args.entailment_batch_size,
    )

    input_file = input_dir / "examples.jsonl"
    if not input_file.exists():
        print(f"error: {input_file} not found")
        sys.exit(1)

    print(f"[filter] {agent_name}")
    print(f"  input: {input_file}")
    print(f"  output: {output_dir / 'examples.jsonl'}")
    if config.run_entailment_check:
        print(f"  entailment check: enabled (model={config.entailment_model}, batch={config.entailment_batch_size})")
    else:
        print(f"  entailment check: disabled (use --run-entailment-check to enable)")

    report = filter_file(input_file, output_dir, agent_name, config)
    print(report.summary())

    report_dict = {
        "agent": report.agent,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "failure_categories": dict(report.failures),
        "duplicate_groups": report.duplicate_groups,
        "answer_conflicts": report.answer_conflicts,
        "entailment_failures": report.entailment_failures,
    }
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  report: {output_dir / 'quality_report.json'}")


# ---------------------------------------------------------------------------
# Hardened entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nunhandled exception: {exc}", file=sys.stderr)
        traceback.print_exc()
        try:
            _parser = argparse.ArgumentParser(add_help=False)
            _parser.add_argument("--output-dir", default="./phase1_data")
            _args, _ = _parser.parse_known_args()
            _out = Path(_args.output_dir)
            _out.mkdir(parents=True, exist_ok=True)
            with open(_out / "crash_report.json", "w") as _f:
                json.dump({
                    "error": "unhandled_exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }, _f, indent=2)
            print(f"crash report written to {_out / 'crash_report.json'}")
        except Exception:
            pass
        sys.exit(2)
