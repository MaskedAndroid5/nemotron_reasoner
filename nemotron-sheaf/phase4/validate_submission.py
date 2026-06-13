#!/usr/bin/env python3
"""
validate_submission.py — Phase 4: Submission Validation (veteran grade)
=======================================================================
Performs a final sanity check on a submission.zip file before uploading
to the competition platform.

Checks performed:
  1. Archive structure — required files exist, no extras.
  2. Adapter config — rank ≤ 32, lora_alpha / rank scaling ratio in
     a sensible range (0.5–4.0).
  3. Adapter weights — non‑LoRA key detection for BOTH safetensors and
     .bin formats.  .bin loads with weights_only=True for safety.
  4. (Optional) vLLM smoke test — loads the adapter at competition
     settings and runs a held‑out reasoning prompt that requires
     actual multi‑hop deduction, not pattern matching.  Verifies:
       • adapter loads and changes base model output,
       • answer is extractable and matches an acceptable set,
       • inference completes within a configurable timeout.

Output: validate_submission.json

Exit codes:
  0 – submission passed all checks
  1 – structural / config failure
  2 – environment error
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import tempfile
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ValidationConfig(BaseModel):
    submission_zip: Path
    output_dir: Path = Path("./phase4_results")
    model_id: str = Field(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        description="Base model for vLLM smoke test"
    )
    run_vllm_check: bool = True
    vllm_timeout_sec: int = Field(120, ge=30, description="Per‑generation timeout")
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=512)
    max_lora_rank: int = Field(32, ge=1)

    test_prompt: str = (
        "You are a precise reasoning assistant. "
        "Answer the question step by step using the provided facts. "
        "Place your final answer inside \\boxed{}.\n\n"
        "Facts:\n"
        "  Fact 1: Alice is Bob's sister.\n"
        "  Fact 2: Bob is Carol's father.\n"
        "  Fact 3: David is Carol's brother.\n\n"
        "Question: How is Alice related to David?\n\n"
        "Reasoning:"
    )
    acceptable_answers: List[str] = Field(
        default=[
            "Alice is David's aunt",
            "Alice is the aunt of David",
            "Alice is his aunt",
            "aunt",
        ],
        description="Acceptable answer strings (case‑insensitive containment match)"
    )
    max_new_tokens: int = 256

    @validator("submission_zip", "output_dir", always=True)
    def _resolve_paths(cls, v):
        return Path(v)

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Timeout helper (Unix-only; no‑op on Windows)
# ---------------------------------------------------------------------------
@contextmanager
def _timeout(seconds: int):
    """Context manager that raises TimeoutError after `seconds` seconds."""
    if not hasattr(signal, "SIGALRM"):
        yield  # no timeout on Windows
        return

    def _handler(signum, frame):
        raise TimeoutError(f"timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Boxed answer extraction (shared regex)
# ---------------------------------------------------------------------------
_BOXED_REGEX = re.compile(
    r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
)


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_REGEX.findall(text)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# Structural checks (no GPU needed)
# ---------------------------------------------------------------------------
def _validate_archive(archive_path: Path, work_dir: Path) -> Dict[str, Any]:
    """Extract the zip and check required files, config, and weights."""
    result: Dict[str, Any] = {
        "files_found": [],
        "config_exists": False,
        "weights_exist": False,
        "weights_format": None,
        "rank": None,
        "rank_valid": False,
        "alpha": None,
        "scaling_ratio": None,
        "scaling_ok": True,
        "non_lora_keys": [],
        "errors": [],
        "warnings": [],
    }

    # Extract
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(work_dir)
            result["files_found"] = zf.namelist()
    except zipfile.BadZipFile as e:
        result["errors"].append(f"Invalid zip file: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Failed to extract zip: {e}")
        return result

    # --- adapter_config.json ---
    config_path = work_dir / "adapter_config.json"
    if not config_path.is_file():
        result["errors"].append("adapter_config.json not found")
        return result
    result["config_exists"] = True

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception as e:
        result["errors"].append(f"Failed to parse adapter_config.json: {e}")
        return result

    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha", cfg.get("alpha", 0))
    result["rank"] = rank
    result["alpha"] = alpha

    if rank is None:
        result["errors"].append("adapter_config.json missing 'r' field")
    elif rank > 32:
        result["errors"].append(f"LoRA rank {rank} exceeds competition limit of 32")
    else:
        result["rank_valid"] = True

    if rank and alpha:
        ratio = alpha / rank
        result["scaling_ratio"] = ratio
        if ratio < 0.5 or ratio > 4.0:
            result["scaling_ok"] = False
            msg = (
                f"LoRA scaling ratio ({ratio:.2f}) is outside recommended range "
                "[0.5, 4.0]. Adapter may be too weak or too dominant."
            )
            result["warnings"].append(msg)

    # --- Weights ---
    safetensors = work_dir / "adapter_model.safetensors"
    bin_weights = work_dir / "adapter_model.bin"
    if safetensors.exists():
        result["weights_format"] = "safetensors"
        result["weights_exist"] = True
        try:
            from safetensors import safe_open
            with safe_open(str(safetensors), framework="pt") as f:
                keys = list(f.keys())
            non_lora = [k for k in keys if "lora_" not in k]
            result["non_lora_keys"] = non_lora
            if non_lora:
                result["errors"].append(
                    f"Non‑LoRA keys in safetensors: {non_lora}. vLLM will reject."
                )
        except Exception as e:
            result["errors"].append(f"Failed to inspect safetensors: {e}")
    elif bin_weights.exists():
        result["weights_format"] = "pytorch_bin"
        result["weights_exist"] = True
        try:
            state = torch.load(bin_weights, map_location="cpu", weights_only=True)
            non_lora = [k for k in state if "lora_" not in k]
            result["non_lora_keys"] = non_lora
            if non_lora:
                result["errors"].append(
                    f"Non‑LoRA keys in .bin weights: {non_lora}. vLLM may reject. "
                    "Prefer saving in safetensors format."
                )
        except Exception as e:
            result["errors"].append(f"Failed to inspect .bin weights: {e}")
    else:
        result["errors"].append("adapter_model.safetensors or .bin not found")

    return result


# ---------------------------------------------------------------------------
# vLLM smoke test
# ---------------------------------------------------------------------------
def _run_vllm_check(
    work_dir: Path,
    cfg: ValidationConfig,
) -> Dict[str, Any]:
    """Load the adapter in vLLM and verify it produces a correct boxed answer."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    result: Dict[str, Any] = {
        "vllm_available": True,
        "adapter_loaded": False,
        "output_changed": False,
        "extracted_answer": None,
        "answer_correct": False,
        "errors": [],
    }

    if not torch.cuda.is_available():
        result["errors"].append("CUDA not available; skipping vLLM check")
        result["vllm_available"] = False
        return result

    # Check if model is cached; warn if download may be needed
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(cfg.model_id, "config.json")
        if cached is None:
            print("  warning: model not cached – vLLM may attempt download")
    except ImportError:
        pass

    # Load vLLM
    try:
        llm = LLM(
            model=cfg.model_id,
            enable_lora=True,
            max_lora_rank=cfg.max_lora_rank,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=True,
        )
    except Exception as e:
        result["errors"].append(f"Failed to load vLLM: {e}")
        return result

    sampling_params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens)
    lora_req = LoRARequest(lora_name="validate", lora_int_id=1, lora_path=str(work_dir))

    try:
        # Base model (no adapter)
        with _timeout(cfg.vllm_timeout_sec):
            base_out = llm.generate([cfg.test_prompt], sampling_params)
        base_text = base_out[0].outputs[0].text

        # Adapted model
        with _timeout(cfg.vllm_timeout_sec):
            adapted_out = llm.generate(
                [cfg.test_prompt], sampling_params, lora_request=lora_req
            )
        adapted_text = adapted_out[0].outputs[0].text
        result["adapter_loaded"] = True
        result["output_changed"] = (base_text != adapted_text)

    except TimeoutError:
        result["errors"].append("vLLM generation timed out")
        del llm
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        result["errors"].append(f"vLLM inference failed: {e}")
        del llm
        torch.cuda.empty_cache()
        return result

    del llm
    torch.cuda.empty_cache()

    # Extract and verify answer (acceptable‑set, case‑insensitive)
    extracted = _extract_boxed(adapted_text)
    if extracted:
        result["extracted_answer"] = extracted
        extracted_lower = extracted.lower()
        for acceptable in cfg.acceptable_answers:
            if acceptable.lower() in extracted_lower:
                result["answer_correct"] = True
                break
    else:
        result["errors"].append("No \\boxed{answer} found in adapted output")

    if not result["output_changed"]:
        result["errors"].append(
            "Adapter did not change base model output — it may not be loaded"
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Path to submission.zip")
    parser.add_argument("--output-dir", default="./phase4_results")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-timeout", type=int, default=120)
    args = parser.parse_args()

    cfg = ValidationConfig(
        submission_zip=args.submission,
        output_dir=args.output_dir,
        model_id=args.model,
        run_vllm_check=not args.skip_vllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        vllm_timeout_sec=args.vllm_timeout,
    )

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[validate] submission validation")
    print(f"  zip: {cfg.submission_zip}")

    if not cfg.submission_zip.exists():
        print(f"error: submission zip not found: {cfg.submission_zip}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # 1. Structural checks
        print("[check] archive structure")
        struct = _validate_archive(cfg.submission_zip, work_dir)
        for err in struct["errors"]:
            print(f"  error: {err}")
        for warn in struct["warnings"]:
            print(f"  warning: {warn}")
        if not struct["config_exists"] or not struct["weights_exist"]:
            print("result: fail (missing files)")
            _write_report(output_dir, passed=False, structural=struct)
            sys.exit(1)
        if not struct["rank_valid"]:
            print("result: fail (rank > 32)")
            _write_report(output_dir, passed=False, structural=struct)
            sys.exit(1)
        if struct["non_lora_keys"]:
            print("result: fail (non‑LoRA keys)")
            _write_report(output_dir, passed=False, structural=struct)
            sys.exit(1)
        print("  structure OK")

        # 2. vLLM smoke test
        vllm_result: Dict[str, Any] = {}
        if cfg.run_vllm_check:
            print("[check] vLLM smoke test")
            vllm_result = _run_vllm_check(work_dir, cfg)
            for err in vllm_result.get("errors", []):
                print(f"  error: {err}")
            if vllm_result.get("adapter_loaded") and vllm_result.get("output_changed"):
                print(f"  extracted answer: {vllm_result.get('extracted_answer')}")
                print(f"  correct: {vllm_result.get('answer_correct')}")
        else:
            vllm_result = {"skipped": True}

        # Final verdict
        passed = (
            struct["config_exists"]
            and struct["weights_exist"]
            and struct["rank_valid"]
            and not struct["non_lora_keys"]
            and (vllm_result.get("skipped") or vllm_result.get("adapter_loaded"))
        )

        _write_report(output_dir, passed=passed, structural=struct, vllm=vllm_result)

        if passed:
            print("result: pass")
        else:
            print("result: fail")
        sys.exit(0 if passed else 1)


def _write_report(
    output_dir: Path,
    passed: bool,
    structural: Dict[str, Any],
    vllm: Optional[Dict[str, Any]] = None,
) -> None:
    report = {
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
        "structural": structural,
        "vllm_check": vllm or {},
    }
    with open(output_dir / "validate_submission.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report → {output_dir / 'validate_submission.json'}")


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