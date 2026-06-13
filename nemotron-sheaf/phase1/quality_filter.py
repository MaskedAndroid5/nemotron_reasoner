#!/usr/bin/env python3
"""
quality_filter.py — Phase 1: reasoning trace quality filter

Reads generated reasoning traces (JSONL) from the Phase 1 data pipeline
and filters out examples that violate any of the following rules:

  1. Structure — missing or malformed required fields.
  2. Boxed answer mismatch — extracted \\boxed{} does not equal ground truth.
  3. XML malformation — unbalanced reasoning tags.
  4. No consistency tag — missing both <compatible> and <incompatible>.
  5. Hallucination — claims introduce too many entities not present in
     the input variables (discourse‑aware, relaxed tolerance).
  6. Token budget — trace length exceeds a configurable token limit.
  7. Near‑duplicate — hash‑based fast path, SequenceMatcher fallback.

Agent‑specific checks are enforced via callable functions (not hard‑coded
strings), making the system extensible without code changes to the core
validation loop.

Failed examples are written to `failures.json` alongside the filtered
`examples.jsonl` and a `quality_report.json` for full provenance tracking.

Usage:
  python quality_filter.py \
      --input-dir phase1_data/logical_deduction \
      --output-dir phase1_data/logical_deduction/filtered
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
        if self.failures_path:
            lines.append(f"  Failures file: {self.failures_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
_BOXED_REGEX = re.compile(
    r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
)

# Discourse markers that are not entities even though capitalised
_DISCOURSE_WORDS: Set[str] = {
    "The", "A", "An", "Therefore", "However", "Thus", "Hence", "So",
    "Conclusion", "Theorem", "Proof", "Lemma", "Definition", "Example",
    "Note", "Consider", "Suppose", "Assume", "Let", "Now", "Next",
    "Finally", "Moreover", "Furthermore", "Indeed", "Also", "Otherwise",
    "Either", "Neither", "Both", "Some", "All", "None", "Any", "Every",
    "Each", "First", "Second", "Third", "Then", "Because", "Since",
    "Nevertheless", "Nonetheless", "Instead", "Rather", "Additionally",
    "Claim", "Reasoning", "Answer", "Question", "Problem", "Solution",
    "Fact", "Statement", "Premise", "Premises",
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
    """
    Extract proper nouns and quoted strings as approximate entities.
    Filters out common discourse markers to avoid false positives
    on sentence‑initial capitalised words.
    """
    proper_nouns = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
    proper_nouns -= _DISCOURSE_WORDS
    quoted = set(re.findall(r'"([^"]+)"', text))
    quoted.update(re.findall(r"'([^']+)'", text))
    return proper_nouns | quoted


def _check_hallucination(
    trace: str, input_text: str, tolerance: int = 5
) -> Tuple[bool, Optional[str]]:
    """
    Return (is_hallucinated, hallucinated_entities_str).
    Allows up to `tolerance` novel entities — reasoning often
    introduces names, examples, or derived concepts naturally.
    """
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
    """Normalise and hash a trace for fast exact‑match duplicate detection."""
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
      2. SequenceMatcher fallback for fuzzy matching.
    """
    trace_hash = _compute_trace_hash(trace)
    for seen in seen_traces[-window_size:]:
        if _compute_trace_hash(seen) == trace_hash:
            return True
    if threshold < 0.95:
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

    # 1. Token budget
    if len(trace.split()) > config.max_tokens * 1.2:
        return False, "token_overflow"

    # 2. XML
    if not _check_xml_wellformed(trace, agent_tags):
        return False, "xml_malformed"

    # 3. Boxed answer
    extracted = _extract_boxed(trace)
    if extracted is None:
        return False, "no_boxed"
    if extracted.strip() != ground_truth.strip():
        return False, "answer_mismatch"

    # 4. Consistency tag (regex, not substring)
    if config.require_consistency_tag:
        has_compat = bool(re.search(r"<compatible\b[^>]*>", trace))
        has_incompat = bool(re.search(r"<incompatible\b[^>]*>", trace))
        if not (has_compat or has_incompat):
            return False, "no_consistency_tag"

    # 5. Minimum claims
    if len(re.findall(r"<claim\b[^>]*>", trace)) < config.min_claims:
        return False, "too_few_claims"

    # 6. Hallucination (relaxed, discourse‑aware)
    input_text = " ".join(str(v) for v in variables.values())
    is_hallucinated, _ = _check_hallucination(
        trace, input_text, tolerance=config.hallucination_tolerance
    )
    if is_hallucinated:
        return False, "hallucination"

    # 7. Agent‑specific checks
    for check_name, check_fn in _get_agent_checks(agent_name).items():
        if not check_fn(trace):
            return False, f"agent_check_{check_name}"

    # 8. Near‑duplicate
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

    with open(input_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                report.failures["json_parse_error"] += 1
                report.total += 1
                failed_examples.append({
                    "error": "json_parse_error",
                    "error_detail": str(e),
                    "line_prefix": line[:200] + ("..." if len(line) > 200 else ""),
                })
                continue

            report.total += 1
            valid, reason = _validate_example(example, config, seen_traces, agent_name)
            if valid:
                report.passed += 1
                seen_traces.append(example["generated_trace"])
                kept.append(example)
            else:
                report.failures[reason] += 1
                failed_examples.append({
                    "example": example,
                    "failure_reason": reason,
                })

            if report.total % 10000 == 0:
                print(f"  progress: {report.total} processed, {report.passed} passed")

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
    )

    input_file = input_dir / "examples.jsonl"
    if not input_file.exists():
        print(f"error: {input_file} not found")
        sys.exit(1)

    print(f"[filter] {agent_name}")
    print(f"  input: {input_file}")
    print(f"  output: {output_dir / 'examples.jsonl'}")

    report = filter_file(input_file, output_dir, agent_name, config)
    print(report.summary())

    report_dict = {
        "agent": report.agent,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "failure_categories": dict(report.failures),
        "duplicate_groups": report.duplicate_groups,
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
        sys.exit(2)