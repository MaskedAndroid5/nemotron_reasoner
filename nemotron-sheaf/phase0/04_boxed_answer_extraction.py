#!/usr/bin/env python3
"""
04_boxed_answer_extraction.py — Phase 0 Gate 4: Answer Extraction Validation
============================================================================
Production‑grade verification that the competition's answer extraction
pipeline correctly handles the output formats our model will produce.

The competition metric uses a multi‑stage parser:
  1. Normalize the response (strip markdown, LaTeX delimiters, whitespace)
     while preserving semantically important content (e.g. dollar amounts).
  2. Search for the LAST occurrence of \boxed{...} with structural
     validation (balanced braces, escaped characters).
  3. If a boxed answer is found, classify its content type and extract it.
  4. If NO boxed answer is found, fall back to the last numeric value
     (including scientific notation).
  5. Compare with ground truth (exact string or relative numerical tolerance).

This implementation supports multiple extraction back‑ends:
  • nemo   – NeMo RL official parser (if installed)
  • regex  – robust regex‑based fallback
  • advanced – extended with fraction recognition and content classification

Output: boxed_answer_extraction.json

Exit codes:
  0 – All primary extraction tests passed
  1 – One or more primary extraction tests failed
  2 – Environment error
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ExtractionConfig(BaseModel):
    output_dir: Path = Path("./phase0_results")
    extractor: str = Field("nemo", description="Extraction back‑end: nemo, regex, advanced")

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Core extraction primitives (regex, structural validation)
# ---------------------------------------------------------------------------

# Matches \boxed{...} with up to 3 levels of nested braces (no recursion)
_BOXED_REGEX = re.compile(
    r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
)

# Fallback numeric: integers, decimals, scientific notation, negative
_FALLBACK_NUMERIC_REGEX = re.compile(
    r'(?:(?<!\d)[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?!\d))'
)


def _validate_brace_balance(text: str) -> bool:
    """Return True if braces are balanced, respecting escaped characters."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '\\' and i + 1 < n:
            # skip escaped character
            i += 2
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def normalize_response(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Strip markdown / LaTeX wrappers while preserving dollar amounts.
    Returns (normalized_text, preservation_map).
    """
    preserved: Dict[str, str] = {}

    # Preserve escaped dollar amounts before stripping $ delimiters
    dollar_pattern = r'\\\$(\d+(?:\.\d+)?)'
    for match in re.finditer(dollar_pattern, text):
        key = f"__DOLLAR_{len(preserved)}__"
        preserved[key] = match.group(1)
        text = text[:match.start()] + key + text[match.end():]

    # Strip markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Strip LaTeX display/inline math
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)

    return text.strip(), preserved


def _restore_preserved(text: str, preserved: Dict[str, str]) -> str:
    """Put back preserved content (e.g. dollar amounts)."""
    for key, value in preserved.items():
        text = text.replace(key, f"${value}")
    return text


# ---------------------------------------------------------------------------
# Extractors (abstract interface)
# ---------------------------------------------------------------------------
class AnswerExtractor:
    """Abstract interface for answer extraction."""
    def extract(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError


class NeMoRLExtractor(AnswerExtractor):
    """Official NeMo RL competition parser."""
    def __init__(self):
        try:
            from nemo_rl.evals.answer_parsing import extract_answer
            self.extract_fn = extract_answer
            self.available = True
        except ImportError:
            self.available = False
            self.extract_fn = None

    def extract(self, text: str) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("NeMo RL not installed")
        try:
            result = self.extract_fn(text)
            return {
                "primary_answer": result,
                "primary_error": None,
                "fallback_answer": None,
                "fallback_error": None,
                "used_fallback": False,
                "warnings": [],
                "method": "nemo_rl_official",
            }
        except Exception as e:
            return {
                "primary_answer": None,
                "primary_error": str(e),
                "fallback_answer": None,
                "fallback_error": None,
                "used_fallback": False,
                "warnings": [f"NeMo RL extraction failed: {e}"],
                "method": "nemo_rl_official",
            }


class RobustRegexExtractor(AnswerExtractor):
    """Regex‑based fallback with structural validation."""
    def extract(self, text: str) -> Dict[str, Any]:
        normalized, preserved = normalize_response(text)
        matches = _BOXED_REGEX.findall(normalized)
        warnings = []

        if matches:
            candidate = matches[-1].strip()
            if not _validate_brace_balance(candidate):
                warnings.append("unbalanced braces in boxed content")
                return {
                    "primary_answer": None,
                    "primary_error": "unbalanced braces",
                    "fallback_answer": None,
                    "fallback_error": None,
                    "used_fallback": False,
                    "warnings": warnings,
                    "method": "regex_robust",
                }
            candidate = _restore_preserved(candidate, preserved)
            return {
                "primary_answer": candidate,
                "primary_error": None,
                "fallback_answer": None,
                "fallback_error": None,
                "used_fallback": False,
                "warnings": warnings,
                "method": "regex_robust",
            }

        # Fallback
        fallback_matches = _FALLBACK_NUMERIC_REGEX.findall(normalized)
        if fallback_matches:
            fallback = fallback_matches[-1].strip()
            return {
                "primary_answer": None,
                "primary_error": "no boxed answer found",
                "fallback_answer": fallback,
                "fallback_error": None,
                "used_fallback": True,
                "warnings": [],
                "method": "regex_robust",
            }

        return {
            "primary_answer": None,
            "primary_error": "no boxed answer found",
            "fallback_answer": None,
            "fallback_error": "no numeric fallback",
            "used_fallback": True,
            "warnings": [],
            "method": "regex_robust",
        }


class AdvancedExtractor(RobustRegexExtractor):
    """Extended extractor with content type classification."""

    FRAC_PATTERN = re.compile(r'\\frac\{([^}]*)\}\{([^}]*)\}')

    def extract(self, text: str) -> Dict[str, Any]:
        result = super().extract(text)
        content = result.get("primary_answer") or result.get("fallback_answer") or ""
        result["content_type"] = self._classify(content)
        return result

    def _classify(self, content: str) -> str:
        if not content:
            return "empty"
        if self.FRAC_PATTERN.search(content):
            return "fraction"
        if re.search(r'[eE][+-]?\d+', content):
            return "scientific_notation"
        if re.search(r'[^\x00-\x7F]', content):
            return "unicode"
        if '\n' in content:
            return "multiline"
        if re.match(r'^[\d.+\-*/^ ]+$', content):
            return "numeric"
        return "text"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_extractor(method: str = "nemo") -> AnswerExtractor:
    if method == "nemo":
        nemo = NeMoRLExtractor()
        if nemo.available:
            print("[extractor] using NeMo RL official parser")
            return nemo
        print("[extractor] NeMo RL not installed; falling back to robust regex")
        return RobustRegexExtractor()
    if method == "advanced":
        print("[extractor] using advanced regex extractor")
        return AdvancedExtractor()
    print("[extractor] using robust regex fallback")
    return RobustRegexExtractor()


# ---------------------------------------------------------------------------
# Expanded test case battery
# ---------------------------------------------------------------------------
def build_test_cases() -> List[Dict[str, Any]]:
    return [
        # --- Primary extraction ---
        {"description": "Standard boxed answer", "input": "\\boxed{42}.", "expected_primary": "42", "expect_fallback": False},
        {"description": "Nested boxed", "input": "\\boxed{\\boxed{184}}.", "expected_primary": "184", "alternatives_primary": ["\\boxed{184}"], "expect_fallback": False},
        {"description": "LaTeX inside boxed", "input": "\\boxed{\\text{Paris}}", "expected_primary": "\\text{Paris}", "expect_fallback": False},
        {"description": "Multiple boxed – last wins", "input": "\\boxed{100}. \\boxed{42}.", "expected_primary": "42", "expect_fallback": False},
        {"description": "Whitespace inside boxed", "input": "  \\boxed{   hello   }  ", "expected_primary": "hello", "expect_fallback": False},
        {"description": "Bold markdown wrapping boxed", "input": "**\\boxed{3.14}**", "expected_primary": "3.14", "expect_fallback": False},
        {"description": "Multi-line reasoning, boxed at end", "input": "Step 1\nStep 2\n\\boxed{final}.", "expected_primary": "final", "expect_fallback": False},
        {"description": "Empty boxed", "input": "\\boxed{}", "expected_primary": "", "expect_fallback": False},
        {"description": "Percent sign inside", "input": "\\boxed{50\\%}", "expected_primary": "50\\%", "expect_fallback": False},
        {"description": "Unicode answer", "input": "\\boxed{🟢}", "expected_primary": "🟢", "expect_fallback": False},
        {"description": "Fraction inside", "input": "\\boxed{\\frac{1}{2}}", "expected_primary": "\\frac{1}{2}", "expect_fallback": False},
        {"description": "Boxed with newlines", "input": "\\boxed{line1\nline2}", "expected_primary": "line1\nline2", "expect_fallback": False},
        {"description": "Inline code inside", "input": "\\boxed{`print(x)`}", "expected_primary": "`print(x)`", "expect_fallback": False},
        {"description": "HTML entities", "input": "\\boxed{A &amp; B}", "expected_primary": "A &amp; B", "expect_fallback": False},
        {"description": "Escaped dollar sign", "input": "\\boxed{\\$50}", "expected_primary": "$50", "expect_fallback": False},
        # --- Structural failures (should fail primary) ---
        {"description": "Unbalanced braces", "input": "\\boxed{foo {bar}", "expected_primary": None, "expect_fallback": False},
        {"description": "Escaped braces (valid)", "input": "\\boxed{\\{answer\\}}", "expected_primary": "\\{answer\\}", "expect_fallback": False},
        # --- Fallback extraction ---
        {"description": "No boxed – fallback integer", "input": "The answer is 42.", "expected_primary": None, "expect_fallback": True, "expected_fallback": "42"},
        {"description": "No boxed – fallback decimal", "input": "The answer is 3.14.", "expected_primary": None, "expect_fallback": True, "expected_fallback": "3.14"},
        {"description": "No boxed – negative", "input": "It is -15.", "expected_primary": None, "expect_fallback": True, "expected_fallback": "-15"},
        {"description": "No boxed – scientific notation", "input": "The result is 1.5e-10.", "expected_primary": None, "expect_fallback": True, "expected_fallback": "1.5e-10"},
        {"description": "No boxed – multiple numbers", "input": "First 100 then 42.", "expected_primary": None, "expect_fallback": True, "expected_fallback": "42"},
        {"description": "No boxed – no number", "input": "No answer here.", "expected_primary": None, "expect_fallback": True, "expected_fallback": None},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./phase0_results")
    parser.add_argument("--extractor", choices=["nemo", "regex", "advanced"], default="nemo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[phase0] boxed answer extraction test")
    extractor = get_extractor(args.extractor)

    test_cases = build_test_cases()
    results = []
    primary_passed = 0
    primary_failed = 0
    fallback_passed = 0
    fallback_failed = 0
    warnings_total = 0

    for tc in test_cases:
        extraction = extractor.extract(tc["input"])
        result = {
            "description": tc["description"],
            "input": tc["input"],
            "expect_fallback": tc["expect_fallback"],
            "extraction": extraction,
            "passed": False,
        }

        if tc["expect_fallback"]:
            expected = tc.get("expected_fallback")
            got = extraction.get("fallback_answer")
            ok = (got is not None and got.strip() == expected.strip()) if expected is not None else (got is None)
            result["expected"] = expected
            result["got"] = got
            result["passed"] = ok
            if ok:
                fallback_passed += 1
                print(f"  [✓] {tc['description']}")
            else:
                fallback_failed += 1
                print(f"  [✗] {tc['description']}  expected: {expected}  got: {got}")
        else:
            expected = tc.get("expected_primary")
            alternatives = tc.get("alternatives_primary", [])
            got = extraction.get("primary_answer")
            ok = False
            if expected is not None:
                ok = (got is not None and
                      (got.strip() == expected.strip() or
                       any(got.strip() == alt.strip() for alt in alternatives)))
            else:
                ok = (got is None)
            result["expected"] = expected
            result["got"] = got
            result["passed"] = ok
            if ok:
                primary_passed += 1
                print(f"  [✓] {tc['description']}")
            else:
                primary_failed += 1
                print(f"  [✗] {tc['description']}  expected: {expected}  got: {got}")

        warnings = extraction.get("warnings", [])
        if warnings:
            warnings_total += len(warnings)
            if args.verbose:
                for w in warnings:
                    print(f"     ⚠ {w}")

        results.append(result)

    total_primary = primary_passed + primary_failed
    total_fallback = fallback_passed + fallback_failed

    print(f"\n  primary: {primary_passed}/{total_primary}  fallback: {fallback_passed}/{total_fallback}  warnings: {warnings_total}")

    # Recommendations
    recommendations = []
    if primary_failed:
        recommendations.append(
            "Primary extraction failures: verify training data always wraps "
            "answers in \\boxed{{answer}}. Add explicit format instructions."
        )
    if fallback_failed:
        recommendations.append(
            "Fallback extraction failures: while the model is trained for "
            "\\boxed{{}}, ensure the fallback parser can handle common numeric "
            "outputs (e.g. scientific notation)."
        )
    if warnings_total:
        recommendations.append(
            "Structural warnings detected (e.g. unbalanced braces). "
            "Review the 'warnings' field in the report for details."
        )
    if not primary_failed and not fallback_failed and not warnings_total:
        recommendations.append("All extraction tests passed. Output format is robust.")

    # Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "extractor_method": args.extractor,
        "primary_tests": total_primary,
        "primary_passed": primary_passed,
        "primary_failed": primary_failed,
        "fallback_tests": total_fallback,
        "fallback_passed": fallback_passed,
        "fallback_failed": fallback_failed,
        "warnings_total": warnings_total,
        "results": results,
        "recommendations": recommendations,
        "summary": {
            "all_primary_passed": primary_failed == 0,
            "all_fallback_passed": fallback_failed == 0,
            "no_warnings": warnings_total == 0,
            "verdict": "PASS" if primary_failed == 0 else "FAIL",
        },
    }
    with open(output_dir / "boxed_answer_extraction.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report → {output_dir / 'boxed_answer_extraction.json'}")

    if primary_failed == 0:
        print("result: pass")
        if fallback_failed:
            print("  (fallback tests failed — investigate if model omits \\boxed{})")
        if warnings_total:
            print(f"  ({warnings_total} structural warning(s) — see report)")
        sys.exit(0)
    else:
        print(f"result: fail ({primary_failed} primary test(s) failed)")
        for rec in recommendations:
            print(f"  → {rec}")
        sys.exit(1)


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
            _parser.add_argument("--output-dir", default="./phase0_results")
            _args, _ = _parser.parse_known_args()
            _out = Path(_args.output_dir)
            _out.mkdir(parents=True, exist_ok=True)
            with open(_out / "crash_report.json", "w") as _f:
                json.dump({
                    "error": "unhandled_exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }, _f, indent=2)
            print(f"crash report written to {_out / 'crash_report.json'}")
        except Exception:
            pass
        sys.exit(2)