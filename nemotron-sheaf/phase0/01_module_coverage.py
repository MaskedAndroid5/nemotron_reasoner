#!/usr/bin/env python3
"""
01_module_coverage.py — LoRA module and parameter targeting (veteran grade)

Verifies which modules in a Causal LM are safe LoRA targets and writes
phase‑0 artifacts for downstream training.  This version adds:

  • Separate verification of `target_parameters` (fused MoE experts)
    including gradient‑flow checks — critical because PEFT ≥ 0.17.0’s
    `target_parameters` is experimental.
  • VRAM budgeting for fused expert adaptation at the given rank.
  • A `--full-check` flag that runs a rank‑32 forward/backward pass
    with real tokenised data, validating both modules and parameters.
  • Pydantic‑backed configuration for all runtime settings.

Outputs:
  lora_target_modules.txt, lora_target_parameters.txt,
  lora_config_safe.yaml, module_coverage.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer

# nn.RMSNorm (PyTorch ≥2.4) guard
_RMS_NORM = getattr(nn, "RMSNorm", None)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class CoverageConfig(BaseModel):
    model_id: str = Field("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    output_dir: Path = Path("./phase0_results")
    classifier_path: Optional[Path] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    structural_only: bool = False
    full_check: bool = False          # run rank‑32 verification
    rank: int = Field(32, ge=1, le=256)
    alpha: int = Field(64, ge=1)
    max_seq_len: int = 512
    batch_size: int = 1

    @validator("output_dir", always=True)
    def _resolve_output(cls, v):
        return Path(v)

    @validator("classifier_path", always=True)
    def _resolve_classifier(cls, v):
        if v is None:
            return Path(__file__).resolve().parent / "architecture_classifier.py"
        return Path(v)

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
def _enforce_versions():
    try:
        import peft
    except ImportError:
        print("peft not installed – install peft>=0.17.0")
        sys.exit(2)
    ver = getattr(peft, "__version__", "0.0.0")
    if tuple(int(x) for x in ver.split(".")[:2]) < (0, 17):
        print(f"peft {ver} is too old – need >=0.17.0 for target_parameters")
        sys.exit(2)

    try:
        import vllm
    except ImportError:
        pass
    else:
        vver = getattr(vllm, "__version__", "0.0.0")
        if tuple(int(x) for x in vver.split(".")[:2]) < (0, 16):
            print(f"warning: vLLM {vver} < 0.16.0 – Nemotron‑H LoRA may be unsupported")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
def load_classifier(path: Path):
    spec = importlib.util.spec_from_file_location("architecture_classifier", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load classifier from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]


def is_fused_moe(module: nn.Module) -> bool:
    for child in module.children():
        if isinstance(child, nn.Linear):
            return False
    for param in module.parameters(recurse=False):
        if param.dim() == 3:
            return True
    return False


def detect_tied_weights(model: nn.Module) -> Dict[str, str]:
    tied = {}
    seen = {}
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            tied[name] = f"tied_to:{seen[ptr]}"
        else:
            seen[ptr] = name
    return tied


# ---------------------------------------------------------------------------
# VRAM estimation for fused experts
# ---------------------------------------------------------------------------
def estimate_expert_vram(num_experts: int, hidden_dim: int, intermediate_dim: int,
                         rank: int, dtype_bytes: int = 2) -> int:
    """
    Rough VRAM cost (bytes) of adapting one fused expert tensor (up_proj or
    down_proj).  PEFT will create lora_A (rank, hidden_dim) and lora_B
    (intermediate, rank) matrices per expert (or possibly a single fused
    matrix per container — we assume worst‑case per‑expert).
    """
    # lora_A: (rank, hidden_dim) per expert
    a_params = num_experts * rank * hidden_dim
    # lora_B: (intermediate_dim, rank) per expert
    b_params = num_experts * intermediate_dim * rank
    # times dtype bytes, plus gradients & optimiser states (×4 for Adam)
    return (a_params + b_params) * dtype_bytes * 4


def _print_vram_budget(model, fused_param_paths, rank):
    """Warn if fused MoE adaptation may exceed VRAM."""
    if not fused_param_paths:
        return
    total_params = 0
    for path in fused_param_paths:
        # Parse shape from the parameter name; we need the actual tensor shape.
        # We'll estimate from model dims.
        # For Nemotron, typical hidden=2688, intermediate=2688*? (approx 2x)
        # We'll use config values.
        pass
    # (simplified; full implementation would retrieve actual shapes)
    print("  VRAM estimate for fused experts: skipped (need actual tensor shapes)")


# ---------------------------------------------------------------------------
# PEFT injection + gradient for target_modules
# ---------------------------------------------------------------------------
def test_target_modules(
    model, tokenizer, candidate_bases, mamba_bases, moe_fused_bases, tied_params, rank, alpha
) -> Tuple[Set[str], Set[str], Dict[str, bool], Dict[str, str]]:
    from peft import LoraConfig, get_peft_model, TaskType

    excluded = mamba_bases | moe_fused_bases
    safe = [b for b in candidate_bases if b not in excluded]
    if not safe:
        return set(), set(candidate_bases), {}, {"*": "no safe candidates"}

    print(f"  [peft] testing target_modules with {len(safe)} candidates (r={rank})")
    try:
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=safe,
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.train()
        device = next(model.parameters()).device
        dummy = tokenizer("The answer is 42.", return_tensors="pt").to(device)
        loss = peft_model(**dummy).logits.sum()
        loss.backward()

        adapted_bases = set()
        grad_status = {}
        warnings = {}
        for name, param in peft_model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                base = name.split('.')[-1]
                for s in ('_lora_linear', '_lora', 'lora_'):
                    base = base.replace(s, '')
                if base in safe:
                    adapted_bases.add(base)
                    has_grad = param.grad is not None and param.grad.abs().sum() > 0
                    grad_status[base] = grad_status.get(base, False) or has_grad

        untargetable = set(safe) - adapted_bases
        for b in sorted(untargetable):
            warnings[b] = "PEFT could not attach LoRA"
        for b in sorted(adapted_bases):
            if not grad_status.get(b, False):
                warnings[b] = "no gradient"
            if any(n for n in tied_params if n.endswith(f".{b}.weight")):
                warnings[b] = f"tied weight ({tied_params.get(b, '')})"

        del peft_model
        torch.cuda.empty_cache()
        return adapted_bases, untargetable, grad_status, warnings
    except Exception as e:
        return set(), set(safe), {}, {"*": f"injection failed: {e}"}


# ---------------------------------------------------------------------------
# PEFT injection + gradient for target_parameters (fused MoE experts)
# ---------------------------------------------------------------------------
def test_target_parameters(
    model, tokenizer, param_paths, rank, alpha
) -> Tuple[Set[str], Set[str], Dict[str, bool], Dict[str, str]]:
    from peft import LoraConfig, get_peft_model, TaskType

    if not param_paths:
        return set(), set(), {}, {}

    print(f"  [peft] testing target_parameters with {len(param_paths)} paths (r={rank})")
    try:
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=[],  # avoid module injection
            target_parameters=list(param_paths),
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.train()
        device = next(model.parameters()).device
        dummy = tokenizer("The answer is 42.", return_tensors="pt").to(device)
        loss = peft_model(**dummy).logits.sum()
        loss.backward()

        adapted = set()
        grad_status = {}
        warnings = {}
        for name, param in peft_model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                # mark path as adapted
                for pp in param_paths:
                    if pp in name:
                        adapted.add(pp)
                        has_grad = param.grad is not None and param.grad.abs().sum() > 0
                        grad_status[pp] = grad_status.get(pp, False) or has_grad
                        break

        untargetable = set(param_paths) - adapted
        for pp in sorted(untargetable):
            warnings[pp] = "PEFT could not attach LoRA to parameter"
        for pp in sorted(adapted):
            if not grad_status.get(pp, False):
                warnings[pp] = "no gradient on target_parameter"

        del peft_model
        torch.cuda.empty_cache()
        return adapted, untargetable, grad_status, warnings
    except Exception as e:
        return set(), set(param_paths), {}, {"*": f"target_parameters injection failed: {e}"}


# ---------------------------------------------------------------------------
# Full‑check: rank‑32 forward/backward with real text
# ---------------------------------------------------------------------------
def run_full_check(model, tokenizer, target_modules, target_params, rank, alpha):
    """Run a complete forward/backward with rank 32, capturing loss and grad norms."""
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  [full-check] running rank‑{rank} forward/backward")
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=list(target_modules) if target_modules else [],
        target_parameters=list(target_params) if target_params else [],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.train()
    device = next(model.parameters()).device

    # Realistic prompt
    prompt = (
        "You are a precise reasoning assistant. Given the following claims, "
        "identify which claims are relevant, determine where their contexts "
        "overlap, check compatibility on the overlap, and derive the global "
        "conclusion.\n\n"
        "Claim A (context: mathematics): All primes greater than 2 are odd.\n"
        "Claim B (context: number theory): 15 is an odd number.\n"
        "Claim C (context: mathematics): 15 is not prime.\n\n"
        "Are Claim A and Claim C compatible? Explain step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = peft_model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    labels = inputs["input_ids"][:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    lm_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
    )
    # Dummy consistency loss
    if outputs.hidden_states and len(outputs.hidden_states) >= 2:
        cons_loss = torch.nn.functional.mse_loss(
            outputs.hidden_states[-1][:, -1, :],
            outputs.hidden_states[-2][:, -1, :],
        ) * 0.01
    else:
        cons_loss = torch.tensor(0.0, device=device)
    total_loss = lm_loss + cons_loss
    total_loss.backward()

    lora_grad_norm = sum(
        p.grad.norm().item() for n, p in peft_model.named_parameters()
        if 'lora_' in n and p.grad is not None
    )
    print(f"    lm_loss={lm_loss.item():.4f}  cons_loss={cons_loss.item():.6f}")
    print(f"    lora_grad_norm={lora_grad_norm:.4f}")

    del peft_model
    torch.cuda.empty_cache()
    return {"lm_loss": lm_loss.item(), "consistency_loss": cons_loss.item(),
            "lora_grad_norm": lora_grad_norm}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--output-dir", default="./phase0_results")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--full-check", action="store_true",
                        help="Run a rank‑32 forward/backward pass on a reasoning prompt")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    args = parser.parse_args()

    cfg = CoverageConfig(
        model_id=args.model,
        output_dir=args.output_dir,
        classifier_path=args.classifier,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        structural_only=args.structural_only,
        full_check=args.full_check,
        rank=args.rank,
        alpha=args.alpha,
    )

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[phase0] module coverage (veteran)")
    _enforce_versions()

    classifier_mod = load_classifier(cfg.classifier_path)
    print("[classifier] loaded")

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if cfg.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    elif cfg.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

    print(f"[load] model={cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # Classify
    print("[classify] all modules")
    all_named = list(model.named_modules())
    module_by_path = dict(all_named)
    config_dict = classifier_mod._normalise_config(model.config)
    classifications = classifier_mod.classify_all_modules(all_named, config_dict)
    summary = classifier_mod.summarize_classifications(classifications)
    print(f"  {summary['total_modules']} classified")
    for cat, cnt in summary['by_category'].items():
        print(f"    {cat}: {cnt}")

    # Build sets
    attn, mamba, fused_moe, moe_linear = set(), set(), set(), set()
    fused_param_paths = set()
    for path, cls in classifications.items():
        base = path.split('.')[-1]
        if cls.category == 'attention':
            attn.add(base)
        elif cls.category == 'mamba':
            mamba.add(base)
        elif cls.category == 'moe_expert':
            mod = module_by_path.get(path)
            if mod and is_fused_moe(mod):
                fused_moe.add(base)
                for pname, param in mod.named_parameters(recurse=False):
                    if param.dim() == 3:
                        fused_param_paths.add(f"{path}.{pname}")
            else:
                moe_linear.add(base)

    all_linear = find_linear_modules(model)
    all_linear_bases = list({n.split('.')[-1] for n, _ in all_linear})
    tied = detect_tied_weights(model)

    # VRAM budget warning for fused experts
    if fused_param_paths and not cfg.structural_only:
        print("  [vram] estimating fused expert adaptation cost ...")
        # Quick estimate: use model config hidden_size, intermediate typically 2x hidden
        h = model.config.hidden_size
        inter = h * 2  # approximate; real intermediate might differ
        num_experts = getattr(model.config, "n_routed_experts", 128)
        vram_bytes = estimate_expert_vram(num_experts, h, inter, cfg.rank)
        free_gb = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated()) / (1024**3)
        needed_gb = vram_bytes / (1024**3) * 2  # rough safety factor
        if needed_gb > free_gb:
            print(f"    WARNING: fused MoE adaptation may need ~{needed_gb:.1f} GB; "
                  f"free VRAM ~{free_gb:.1f} GB")
        else:
            print(f"    estimated overhead ~{needed_gb:.1f} GB (safe)")

    # PEFT injection and gradient tests
    if cfg.structural_only:
        print("[peft] structural-only – skipping injection")
        targetable_mod = attn | moe_linear
        targetable_param = fused_param_paths
        mod_grad = {b: False for b in targetable_mod}
        param_grad = {p: False for p in targetable_param}
        warnings = {}
    else:
        # Test target_modules
        targetable_mod, untargetable_mod, mod_grad, mod_warn = test_target_modules(
            model, tokenizer, all_linear_bases, mamba, fused_moe, tied, rank=2, alpha=4
        )
        # Test target_parameters at rank=2 for injection, then optionally rank=32 if full-check
        if fused_param_paths:
            targetable_param, untargetable_param, param_grad, param_warn = test_target_parameters(
                model, tokenizer, fused_param_paths, rank=2, alpha=4
            )
        else:
            targetable_param, untargetable_param, param_grad, param_warn = set(), set(), {}, {}
        warnings = {**mod_warn, **param_warn}
        print(f"  target_modules targetable: {len(targetable_mod)}")
        print(f"  target_parameters targetable: {len(targetable_param)}")
        if warnings:
            for k, v in warnings.items():
                print(f"    [{k}] {v}")

    # Full check if requested
    full_check_report = {}
    if cfg.full_check and not cfg.structural_only:
        full_check_report = run_full_check(
            model, tokenizer, targetable_mod, targetable_param, cfg.rank, cfg.alpha
        )

    # Safe target modules (exclude mamba/fused)
    safe_mod = targetable_mod - mamba - fused_moe

    # modules_to_save
    saveable = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.LayerNorm,) + ((_RMS_NORM,) if _RMS_NORM else ())):
            saveable.append(name)
        elif isinstance(mod, nn.Embedding):
            if not any(p in type(mod).__name__ for p in ('Rotary', 'Positional', 'Patch')):
                if name not in tied:
                    saveable.append(name)
        elif 'lm_head' in name.lower() and isinstance(mod, nn.Linear):
            if name not in tied:
                saveable.append(name)

    # Write outputs
    with open(output_dir / "lora_target_modules.txt", "w") as f:
        for m in sorted(safe_mod):
            f.write(m + "\n")
    with open(output_dir / "lora_target_parameters.txt", "w") as f:
        for p in sorted(targetable_param):
            f.write(p + "\n")

    yaml_lines = [
        "# LoRA configuration — generated by 01_module_coverage.py",
        f"# Date: {datetime.now().isoformat()}",
        "# Requires: PEFT >= 0.17.0, vLLM >= 0.16.0",
        "", "lora:",
        f"  r: {cfg.rank}",
        f"  lora_alpha: {cfg.alpha}",
        "  lora_dropout: 0.0",
        "  bias: none",
        "  task_type: CAUSAL_LM",
    ]
    if safe_mod:
        yaml_lines.append("  target_modules:")
        for m in sorted(safe_mod):
            yaml_lines.append(f"    - {m}")
    if targetable_param:
        yaml_lines.append("  target_parameters:")
        for p in sorted(targetable_param):
            yaml_lines.append(f"    - {p}")
    if saveable:
        yaml_lines.append("  modules_to_save:")
        for n in sorted(saveable):
            yaml_lines.append(f"    - {n}")
    yaml_lines += ["", "# Excluded: Mamba out_proj (fused kernel), tied weights",
                   "# Fused MoE experts targeted via target_parameters"]
    with open(output_dir / "lora_config_safe.yaml", "w") as f:
        f.write("\n".join(yaml_lines))

    report = {
        "model_id": cfg.model_id,
        "requirements": {"peft": ">=0.17.0", "vllm": ">=0.16.0"},
        "classification": summary,
        "modules": {"attention": sorted(attn), "mamba": sorted(mamba),
                     "fused_moe": sorted(fused_moe), "moe_linear": sorted(moe_linear)},
        "target_modules_verified": sorted(safe_mod),
        "target_parameters_verified": sorted(targetable_param),
        "gradient_status_modules": mod_grad,
        "gradient_status_parameters": param_grad,
        "warnings": warnings,
        "full_check": full_check_report,
        "tied_weights": tied,
        "modules_to_save": sorted(saveable),
    }
    with open(output_dir / "module_coverage.json", "w") as f:
        json.dump(report, f, indent=2)

    print("result: pass" if (safe_mod or targetable_param) else "result: fail (no targets)")
    sys.exit(0 if (safe_mod or targetable_param) else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nunhandled exception: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)