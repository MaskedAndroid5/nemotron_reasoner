#!/usr/bin/env python3
"""
error_analysis.py — Phase 3: reasoning error categorisation

Parses model-generated reasoning traces, locates sheaf tags, and buckets
every failure into a specific substep so you know **where** the model
struggles – not just that it got the answer wrong.

Failure taxonomy (aligned with the sheaf pipeline):
  • missing_claim          – fewer than min_claims <claim> tags found
  • missing_overlap        – no <overlap> tag present
  • no_boxed               – no \\boxed{…} in the output
  • answer_mismatch        – boxed answer ≠ ground truth
  • no_consistency_tag     – neither <compatible> nor <incompatible> found
  • xml_malformed          – unbalanced reasoning tags
  • agent_specific         – missing required agent‑specific tag (e.g. <assert>)

Output: error_report.json

Usage:
  python error_analysis.py \\
      --input phase1_data/logical_deduction/examples.jsonl \\
      --output phase3_results/error_report.json \\
      --agent logical_deduction
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reasoning_taxonomy import get_agent
from sheaf_consistency_loss import TagPositionExtractor
from transformers import AutoTokenizer  # needed for TagPositionExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")

# Agent‑specific checks from quality_filter.py (duplicated here for standalone use)
AGENT_SPECIFIC_CHECKS: Dict[str, str] = {
    "code_reasoning": ("<assert", "missing_assert"),
    "iterative_state_transition": ("<claim step=", "missing_step_attribute"),
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _extract_boxed(text: str) -> Optional[str]:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def _check_xml_wellformed(text: str, tags: List[str]) -> bool:
    for tag in tags:
        opens = len(re.findall(rf"<{tag}\b[^>]*>", text))
        closes = len(re.findall(rf"</{tag}>", text))
        if opens != closes:
            return False
    return True


def analyse_example(
    example: Dict[str, Any],
    agent_name: str,
    extractor: TagPositionExtractor,
) -> Tuple[bool, List[str]]:
    """
    Return (is_correct, list_of_failure_reasons).
    An empty list means the example passed all checks.
    """
    trace = example.get("generated_trace", "")
    ground_truth = example.get("ground_truth", "")
    agent = get_agent(agent_name)
    tags = list(agent.output_tags.keys())
    reasons: List[str] = []

    # 1. Minimum claims
    n_claims = len(extractor.find_positions(
        torch.tensor(extractor.tokenizer.encode(trace)), extractor.CLAIM_TAGS
    ))
    if n_claims < 1:
        reasons.append("missing_claim")

    # 2. Overlap
    n_overlaps = len(extractor.find_positions(
        torch.tensor(extractor.tokenizer.encode(trace)), extractor.OVERLAP_TAGS
    ))
    if n_overlaps < 1:
        reasons.append("missing_overlap")

    # 3. Boxed answer
    extracted = _extract_boxed(trace)
    if extracted is None:
        reasons.append("no_boxed")
    elif extracted.strip() != ground_truth.strip():
        reasons.append("answer_mismatch")

    # 4. XML well‑formedness
    if not _check_xml_wellformed(trace, tags):
        reasons.append("xml_malformed")

    # 5. Consistency tag
    has_compat = "<compatible>" in trace or "<incompatible" in trace
    if not has_compat:
        reasons.append("no_consistency_tag")

    # 6. Agent‑specific
    if agent_name in AGENT_SPECIFIC_CHECKS:
        required_tag, error_tag = AGENT_SPECIFIC_CHECKS[agent_name]
        if required_tag not in trace:
            reasons.append(error_tag)

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL file with reasoning traces")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument("--agent", required=True, help="Agent name (e.g. logical_deduction)")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Model ID for tokenizer (used by tag extractor)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[error_analysis] agent={args.agent}  input={input_path}")

    # Load tokenizer and extractor
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    extractor = TagPositionExtractor(tokenizer)

    examples = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    total = len(examples)
    passed = 0
    failure_counter: Counter = Counter()
    failed_examples: List[Dict[str, Any]] = []

    for ex in examples:
        ok, reasons = analyse_example(ex, args.agent, extractor)
        if ok:
            passed += 1
        else:
            for r in reasons:
                failure_counter[r] += 1
            failed_examples.append({
                "variables": ex.get("variables", {}),
                "ground_truth": ex.get("ground_truth", ""),
                "generated_trace": ex.get("generated_trace", "")[:500],  # snippet
                "failure_reasons": reasons,
            })

    report = {
        "agent": args.agent,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total > 0 else 0.0,
        "failure_categories": dict(failure_counter),
        "failed_examples": failed_examples,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  accuracy: {report['accuracy']:.3f}  ({passed}/{total})")
    if failure_counter:
        print("  failures:")
        for reason, count in failure_counter.most_common():
            print(f"    {reason}: {count}")
    print(f"  report: {output_path}")


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