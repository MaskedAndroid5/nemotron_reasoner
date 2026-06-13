#!/usr/bin/env python3
"""
05_integration_smoke_test.py — end‑to‑end pipeline smoke test
=============================================================
Validates the full LoRA → training → vLLM → extraction pipeline
**before** committing hours of GPU time to Phase 2.

Checks performed (in order, stops at first failure):
  1. Environment – PEFT ≥ 0.17.0, vLLM ≥ 0.16.0, CUDA, VRAM estimate.
  2. Config – loads and validates lora_config_safe.yaml (Gate 1 output).
  3. Adapter creation – builds rank‑32 adapter with target_modules +
     target_parameters.
  4. Training dry‑run – three forward/backward passes with Adam optimizer.
     Verifies:
       • LoRA parameters receive non‑zero gradients (checked BEFORE zero_grad).
       • Loss trend is non‑increasing over three steps (relaxed: flat is OK).
  5. Adapter export & deep structure validation – safetensors keys,
     config.json, non‑LoRA keys, minimum file size.
  6. PEFT inference – loads the adapter, confirms LoRA layers exist,
     and verifies adapter changes base model output on a known‑answer problem.
  7. vLLM inference (competition params) – gpu_memory_utilization=0.85,
     max_model_len=8192, max_lora_rank=32.  Explicitly validates adapter
     load, then verifies output changes and \\boxed{answer} matches
     expected value.
  8. Extraction – uses NeMo RL parser (or self‑contained regex fallback)
     to validate answer format.
  9. Stress – runs a realistic long‑form reasoning prompt to check VRAM
     headroom.

Exit codes:
  0 – all checks passed; safe to start full training
  1 – pipeline failure
  2 – environment error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class SmokeTestConfig(BaseModel):
    model_id: str = Field("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    phase0_dir: Path = Path("./phase0_results")
    output_dir: Path = Path("./phase0_results")
    lora_config_yaml: Path = Path("./phase0_results/lora_config_safe.yaml")

    rank: int = Field(32, ge=1, le=32)
    alpha: int = Field(64, ge=1)
    learning_rate: float = Field(1e-4, gt=0.0)

    # vLLM competition parameters
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=512)
    max_lora_rank: int = Field(32, ge=1)

    test_prompt: str = Field(
        default=(
            "You are a precise mathematical reasoner. "
            "Solve the following problem step by step and place your final "
            "answer inside \\boxed{}.\n\n"
            "Problem: If x + 3 = 7, what is x?\n\n"
            "Reasoning:"
        ),
        description="Simple known‑answer prompt"
    )
    expected_answer: str = Field("4", description="Expected boxed answer")
    max_new_tokens: int = Field(128, ge=1)

    skip_stress: bool = Field(False, description="Skip max‑length stress test")

    @validator("output_dir", "phase0_dir", "lora_config_yaml", always=True)
    def _resolve_paths(cls, v):
        return Path(v)

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
def _check_environment() -> float:
    try:
        import peft
    except ImportError:
        print("error: PEFT not installed (pip install peft>=0.17.0)")
        sys.exit(2)
    peft_ver = getattr(peft, "__version__", "0.0.0")
    if tuple(int(x) for x in peft_ver.split(".")[:2]) < (0, 17):
        print(f"error: PEFT {peft_ver} < 0.17.0 – target_parameters unsupported")
        sys.exit(2)

    try:
        import vllm
    except ImportError:
        print("error: vLLM not installed (pip install vllm>=0.16.0)")
        sys.exit(2)
    vllm_ver = getattr(vllm, "__version__", "0.0.0")
    if tuple(int(x) for x in vllm_ver.split(".")[:2]) < (0, 16):
        print(f"error: vLLM {vllm_ver} < 0.16.0 – Nemotron‑H LoRA unsupported")
        sys.exit(2)

    if not torch.cuda.is_available():
        print("error: CUDA not available")
        sys.exit(2)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[env] {gpu_name} ({total_vram:.1f} GB)")
    if total_vram < 48:
        print("warning: GPU < 48 GB; may OOM at competition settings")
    return total_vram


# ---------------------------------------------------------------------------
# Adapter structure validation (deep)
# ---------------------------------------------------------------------------
def _validate_adapter_deep(adapter_dir: Path, expected_rank: int) -> Dict[str, Any]:
    errors = []
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return {"valid": False, "errors": ["adapter_config.json missing"]}

    cfg = json.loads(cfg_path.read_text())
    if cfg.get("r") != expected_rank:
        errors.append(f"rank mismatch: {cfg.get('r')} vs {expected_rank}")

    st = adapter_dir / "adapter_model.safetensors"
    if not st.exists():
        st = adapter_dir / "adapter_model.bin"
    if not st.exists():
        return {"valid": False, "errors": ["no adapter weights found"]}

    file_size_mb = os.path.getsize(st) / (1024**2)
    if file_size_mb < 0.5:
        errors.append(f"adapter file suspiciously small ({file_size_mb:.2f} MB)")

    if st.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(st), framework="pt") as f:
            keys = list(f.keys())
        non_lora = [k for k in keys if "lora_" not in k]
        if non_lora:
            errors.append(f"non‑LoRA keys in safetensors: {non_lora}")

    return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Self‑contained answer extractor (no import from phase0)
# ---------------------------------------------------------------------------
def _get_extractor():
    """Return an extractor – NeMo RL if available, else inline regex."""
    try:
        from nemo_rl.evals.answer_parsing import extract_answer as nemo_extract
        print("[extractor] NeMo RL official parser")

        class _NeMo:
            def extract(self, text: str) -> Dict[str, Any]:
                try:
                    ans = nemo_extract(text)
                    return {"primary_answer": ans, "primary_error": None, "used_fallback": False, "warnings": []}
                except Exception as e:
                    return {"primary_answer": None, "primary_error": str(e), "used_fallback": False, "warnings": [str(e)]}
        return _NeMo()
    except ImportError:
        print("[extractor] inline regex fallback")

    _BOXED = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')
    _NUMERIC = re.compile(r'(?:(?<!\d)[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?!\d))')

    class _Regex:
        def extract(self, text: str) -> Dict[str, Any]:
            for m in _BOXED.findall(text):
                return {"primary_answer": m.strip(), "primary_error": None, "used_fallback": False, "warnings": []}
            for m in _NUMERIC.findall(text):
                return {"primary_answer": None, "primary_error": "no boxed", "fallback_answer": m.strip(), "used_fallback": True, "warnings": []}
            return {"primary_answer": None, "primary_error": "no boxed or numeric", "fallback_answer": None, "used_fallback": True, "warnings": []}
    return _Regex()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--phase0-dir", default="./phase0_results")
    parser.add_argument("--output-dir", default="./phase0_results")
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()

    cfg = SmokeTestConfig(
        model_id=args.model,
        phase0_dir=args.phase0_dir,
        output_dir=args.output_dir,
        lora_config_yaml=Path(args.phase0_dir) / "lora_config_safe.yaml",
        skip_stress=args.skip_stress,
    )

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    report: Dict[str, Any] = {
        "passed": False,
        "model": cfg.model_id,
        "checks": {},
        "errors": [],
        "warnings": [],
        "timestamp": datetime.now().isoformat(),
    }

    print("[phase0] integration smoke test")
    try:
        # 1. Environment
        _check_environment()
        report["checks"]["environment"] = "ok"

        # 2. Config
        safe_yaml = cfg.lora_config_yaml
        if not safe_yaml.exists():
            print(f"error: {safe_yaml} not found – run 01_module_coverage.py first")
            sys.exit(2)
        lora_cfg = yaml.safe_load(safe_yaml.read_text())["lora"]
        target_modules = lora_cfg.get("target_modules", [])
        target_parameters = lora_cfg.get("target_parameters", [])
        print(f"[config] rank={cfg.rank}, alpha={cfg.alpha}")
        print(f"  target_modules: {target_modules or '(none)'}")
        print(f"  target_parameters: {target_parameters or '(none)'}")
        report["checks"]["config"] = "ok"

        # 3. Model & tokenizer
        print(f"[load] {cfg.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        report["checks"]["model_load"] = "ok"

        # 4. Adapter creation
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            r=cfg.rank, lora_alpha=cfg.alpha,
            target_modules=target_modules, target_parameters=target_parameters,
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.train()
        print("[peft] adapter created")
        report["checks"]["adapter_creation"] = "ok"

        # 5. Training dry‑run (three steps for stable signal)
        print("[train] running three‑step training check")
        from torch.optim import Adam

        optimizer = Adam(
            (p for n, p in peft_model.named_parameters() if "lora_" in n),
            lr=cfg.learning_rate,
        )
        device = next(model.parameters()).device
        inputs = tokenizer(cfg.test_prompt, return_tensors="pt").to(device)

        losses = []
        lora_grad_norm = 0.0

        for step_idx in range(3):
            peft_model.zero_grad()
            out = peft_model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
                inputs["input_ids"][:, 1:].reshape(-1),
            )
            loss.backward()

            if step_idx == 0:
                # Capture gradient norm BEFORE any zero_grad
                lora_grad_norm = sum(
                    p.grad.norm().item() for n, p in peft_model.named_parameters()
                    if "lora_" in n and p.grad is not None
                )
                if lora_grad_norm == 0.0:
                    report["errors"].append("no gradients on LoRA parameters after backward()")
                    report["checks"]["gradients"] = "no_gradients"
                else:
                    report["checks"]["gradients"] = "ok"

            optimizer.step()
            losses.append(loss.item())
            print(f"    step {step_idx+1}: loss={loss.item():.4f}")

        # Analyse loss trend (relaxed)
        if len(losses) >= 2:
            avg_early = sum(losses[:2]) / 2
            avg_late = losses[-1]
            loss_trend = avg_early - avg_late
            print(f"  loss trend (early→late): {loss_trend:.6f}")

            if loss_trend >= 0:
                report["checks"]["training"] = "ok"
                if loss_trend < 1e-6:
                    report["warnings"].append("loss trend flat (learning rate may be too low)")
            else:
                report["checks"]["training"] = "loss_increasing"
                report["errors"].append(f"loss increased over 3 steps: trend={loss_trend:.6f}")
        else:
            report["checks"]["training"] = "ok"  # single step? shouldn't happen

        del optimizer

        # 6. Save & validate adapter
        adapter_dir = output_dir / "smoke_test_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_dir))
        del peft_model
        torch.cuda.empty_cache()
        print(f"[save] adapter saved to {adapter_dir}")

        valid = _validate_adapter_deep(adapter_dir, cfg.rank)
        if not valid["valid"]:
            print("error: adapter structure invalid")
            for err in valid["errors"]:
                print(f"  {err}")
                report["errors"].append(err)
            sys.exit(1)
        print("  adapter structure OK")
        report["checks"]["adapter_structure"] = "ok"

        # 7. PEFT inference
        from peft import PeftModel
        print("[peft] inference check")
        peft_inf = PeftModel.from_pretrained(model, str(adapter_dir))
        peft_inf.eval()

        # Verify LoRA layers exist
        lora_layers = [n for n, _ in peft_inf.named_modules() if "lora_" in n]
        if not lora_layers:
            report["errors"].append("LoRA layers not found in PeftModel after loading")
            report["checks"]["peft_load"] = "no_lora_layers"
        else:
            print(f"  loaded {len(lora_layers)} LoRA layers")
            report["checks"]["peft_load"] = "ok"

        with torch.no_grad():
            base_ids = model.generate(
                **tokenizer(cfg.test_prompt, return_tensors="pt").to(device),
                max_new_tokens=cfg.max_new_tokens, do_sample=False,
            )
            adapted_ids = peft_inf.generate(
                **tokenizer(cfg.test_prompt, return_tensors="pt").to(device),
                max_new_tokens=cfg.max_new_tokens, do_sample=False,
            )
        base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
        adapted_text = tokenizer.decode(adapted_ids[0], skip_special_tokens=True)
        changed = base_text != adapted_text
        print(f"  base: {base_text[-120:]}")
        print(f"  adapted: {adapted_text[-120:]}")
        if not changed:
            report["errors"].append("PEFT adapter did not change output")
        report["checks"]["peft_inference"] = "ok" if changed else "no_change"
        del peft_inf
        torch.cuda.empty_cache()

        # 8. vLLM inference at competition parameters
        print("[vllm] loading with competition settings")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        try:
            llm = LLM(
                model=cfg.model_id, enable_lora=True,
                max_lora_rank=cfg.max_lora_rank,
                max_model_len=cfg.max_model_len,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                trust_remote_code=True,
            )
        except torch.cuda.OutOfMemoryError:
            print("error: OOM during vLLM load")
            report["errors"].append("vLLM OOM at competition settings")
            sys.exit(1)

        sampling_params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens, logprobs=1)

        # --- Explicit adapter load validation ---
        lora_req = LoRARequest(lora_name="smoke", lora_int_id=1, lora_path=str(adapter_dir))
        try:
            print("[vllm] testing adapter load")
            test_gen = llm.generate(["test"], SamplingParams(temperature=0.0, max_tokens=1), lora_request=lora_req)
            if not test_gen or not test_gen[0].outputs:
                raise RuntimeError("vLLM returned no outputs")
            report["checks"]["vllm_adapter_load"] = "ok"
            print("  adapter loaded successfully")
        except Exception as e:
            report["errors"].append(f"vLLM adapter load failed: {e}")
            report["checks"]["vllm_adapter_load"] = "failed"
            sys.exit(1)

        # Base model inference (no adapter)
        base_out = llm.generate([cfg.test_prompt], sampling_params)
        base_text_vllm = base_out[0].outputs[0].text

        # Adapted inference
        adapted_out = llm.generate([cfg.test_prompt], sampling_params, lora_request=lora_req)
        adapted_text_vllm = adapted_out[0].outputs[0].text

        vllm_changed = base_text_vllm != adapted_text_vllm
        print(f"  vLLM base: {base_text_vllm[:150]}")
        print(f"  vLLM adapted: {adapted_text_vllm[:150]}")
        if not vllm_changed:
            report["errors"].append("vLLM adapter did not change output")
        report["checks"]["vllm_inference"] = "ok" if vllm_changed else "no_change"

        if not adapted_text_vllm or len(adapted_text_vllm.strip()) == 0:
            report["errors"].append("vLLM adapted generation is empty")
            report["checks"]["vllm_generation"] = "empty"
        elif len(adapted_text_vllm.split()) < 2:
            report["warnings"].append("vLLM adapted generation is very short")

        # 9. Answer extraction
        extractor = _get_extractor()
        extraction = extractor.extract(adapted_text_vllm)
        extracted_answer = extraction.get("primary_answer")
        if extracted_answer is None:
            report["errors"].append("no boxed answer in vLLM output")
            report["checks"]["answer_extraction"] = "no_boxed"
        else:
            correct = extracted_answer.strip() == cfg.expected_answer
            print(f"  extracted: {extracted_answer}  correct={correct}")
            if not correct:
                report["errors"].append(f"wrong answer: '{extracted_answer}' vs '{cfg.expected_answer}'")
            report["checks"]["answer_extraction"] = "ok" if correct else "wrong_answer"

        # 10. Stress test (realistic long‑form prompt)
        if not cfg.skip_stress:
            print("[stress] realistic long‑form reasoning prompt")
            stress_prompt = (
                "You are a mathematical reasoning assistant.\n\n"
                "Prove the following theorem step by step, considering all cases:\n"
                "Theorem: For all positive integers n, the sum of the first n odd numbers equals n^2.\n\n"
                "Proof:\n" +
                "\n".join(
                    f"Case {i}: Consider n={i}. "
                    f"The sum of the first {i} odd numbers is: "
                    f"{' + '.join(str(2*j+1) for j in range(i))} = {i*i}. "
                    f"This follows from the inductive hypothesis."
                    for i in range(1, 50)
                ) +
                "\n\n"
                "Now extend the proof to negative integers and zero where applicable. "
                "Identify edge cases and provide a complete, rigorous proof.\n"
                "Place your final conclusion in \\boxed{}."
            )
            try:
                _ = llm.generate([stress_prompt], sampling_params)
                report["checks"]["stress_test"] = "ok"
                print("  stress test passed")
            except torch.cuda.OutOfMemoryError:
                report["errors"].append("OOM during stress test")
                report["checks"]["stress_test"] = "oom"
                print("warning: OOM at max length")
        else:
            report["checks"]["stress_test"] = "skipped"

        del llm
        torch.cuda.empty_cache()

        # Finalize
        elapsed = time.perf_counter() - t_start
        report["passed"] = len(report["errors"]) == 0
        report["elapsed_sec"] = round(elapsed, 1)
        report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        with open(output_dir / "integration_smoke_test.json", "w") as f:
            json.dump(report, f, indent=2)

        if report["passed"]:
            print(f"result: pass (elapsed={elapsed:.1f}s)")
        else:
            print("result: fail")
            for err in report["errors"]:
                print(f"  - {err}")
        sys.exit(0 if report["passed"] else 1)

    except Exception as exc:
        print(f"\nunhandled exception: {exc}", file=sys.stderr)
        traceback.print_exc()
        report["errors"].append(f"unhandled exception: {exc}")
        report["passed"] = False
        with open(output_dir / "integration_smoke_test.json", "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(2)


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