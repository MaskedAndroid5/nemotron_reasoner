#!/usr/bin/env python3
"""
02_hidden_state_extraction.py — hidden state accessibility verification

Verifies that intermediate hidden states can be extracted from attention
layers during a forward pass.  This is a hard requirement for the
sheaf‑consistency auxiliary loss in Phase 2, which needs to locate
hidden states at specific claim positions and project them into a shared
subspace.

The script:
  1. Identifies attention layers (class‑name inspection, with a fallback
     that avoids hooking leaf Linear submodules).
  2. Registers forward hooks on those layers.
  3. Runs a structured reasoning prompt through the model.
  4. Records hidden states at each hooked layer across generation steps.
  5. Produces a report with per‑layer shapes, VRAM peak, and recommended
     layers for the auxiliary loss (deepest viable layers, in model order).

Output: hidden_state_extraction.json

Exit codes:
  0 – extraction successful on at least one attention layer
  1 – extraction failed (blocking – auxiliary loss not feasible)
  2 – environment error
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = textwrap.dedent("""\
    You are a precise reasoning assistant. Given the following claims,
    identify which claims are relevant, determine where their contexts
    overlap, check compatibility on the overlap, and derive the global
    conclusion.

    Claim A (context: mathematics): All primes greater than 2 are odd.
    Claim B (context: number theory): 15 is an odd number.
    Claim C (context: mathematics): 15 is not prime.

    Are Claim A and Claim C compatible? Explain step by step.
""")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    elif load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attention layer identification
# ---------------------------------------------------------------------------
def find_attention_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return (full_path, module) for modules whose class name contains
    known attention patterns.  Falls back to path‑based matching, but
    only for container modules (modules with children) to avoid hooking
    individual Linear projections.
    """
    attention_class_patterns = [
        "Attention", "SelfAttention", "FlashAttention",
        "MultiHeadAttention", "GroupedQueryAttention", "GQA",
        "NemotronHAttention",
    ]

    layers = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if any(pattern in class_name for pattern in attention_class_patterns):
            layers.append((name, module))

    if not layers:
        print("  no attention layers found by class name – trying path fallback")
        for name, module in model.named_modules():
            if ("attention" in name.lower() or "attn" in name.lower()):
                # Only hook container modules, not leaf Linear layers
                if any(True for _ in module.children()):
                    layers.append((name, module))

    return layers


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------
def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    attention_layers: List[Tuple[str, nn.Module]],
    test_prompt: str,
    max_new_tokens: int = 64,
) -> Tuple[Dict[str, List[torch.Tensor]], float, Dict[str, Any]]:
    """
    Register forward hooks on attention layers, run generation, and
    collect hidden states at each hook invocation.

    Returns:
      extracted: dict mapping layer_path -> list of tensors (one per call)
      vram_peak_gb: peak GPU memory during generation
      metadata: dict with generation info
    """
    extracted: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            elif isinstance(output, torch.Tensor):
                hs = output
            else:
                extracted.setdefault(layer_name, []).append(
                    ("unexpected_type", type(output))
                )
                return
            extracted.setdefault(layer_name, []).append(hs.detach().clone())
        return hook_fn

    for path, module in attention_layers:
        hook = module.register_forward_hook(make_hook(path))
        hooks.append(hook)

    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=False,
            return_dict_in_generate=True,
        )

    vram_peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    for hook in hooks:
        hook.remove()

    generated_ids = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metadata = {
        "input_length": input_length,
        "generated_tokens": len(generated_ids),
        "total_sequence_length": input_length + len(generated_ids),
        "generated_text": generated_text,
        "num_hooked_layers": len(attention_layers),
        "layers_with_output": len(extracted),
    }
    return extracted, vram_peak_gb, metadata


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_extraction(
    extracted: Dict[str, List[torch.Tensor]],
    attention_layers: List[Tuple[str, nn.Module]],
    metadata: Dict[str, Any],
    vram_peak_gb: float,
) -> Dict[str, Any]:
    """Produce a structured analysis of the extraction results."""
    analysis: Dict[str, Any] = {
        "extraction_possible": False,
        "total_layers_hooked": len(attention_layers),
        "layers_extracted": len(extracted),
        "layers_failed": len(attention_layers) - len(extracted),
        "per_layer": {},
        "vram_peak_gb": vram_peak_gb,
        "recommended_layers": [],
        "warnings": [],
    }

    hooked_paths = {path for path, _ in attention_layers}

    for path, tensors in extracted.items():
        actual = [t for t in tensors if isinstance(t, torch.Tensor)]
        if not actual:
            analysis["per_layer"][path] = {
                "extracted": False,
                "error": "no valid tensors captured",
            }
            continue
        shapes = [tuple(t.shape) for t in actual]
        analysis["per_layer"][path] = {
            "extracted": True,
            "num_calls": len(actual),
            "shapes": [list(s) for s in shapes],
            "hidden_size": shapes[0][-1] if shapes else None,
        }

    for path in hooked_paths:
        if path not in extracted:
            analysis["per_layer"][path] = {
                "extracted": False,
                "error": (
                    "hook registered but no output captured – "
                    "layer may be bypassed by fused operations or "
                    "pipeline parallelism"
                ),
            }
            analysis["warnings"].append(
                f"layer '{path}' produced no output"
            )

    # Determine viability and recommend deepest viable layers
    ordered_viable = [
        path for path, _ in attention_layers
        if analysis["per_layer"].get(path, {}).get("extracted")
    ]
    if ordered_viable:
        analysis["extraction_possible"] = True
        analysis["recommended_layers"] = ordered_viable[-3:]
        if len(ordered_viable) < 3:
            analysis["warnings"].append(
                f"only {len(ordered_viable)} layers extracted; "
                "use all available layers for auxiliary loss"
            )
    else:
        analysis["warnings"].append(
            "CRITICAL: no hidden states extracted from any attention layer – "
            "sheaf‑consistency auxiliary loss cannot be implemented"
        )

    if vram_peak_gb > 40:
        analysis["warnings"].append(
            f"peak VRAM {vram_peak_gb:.1f} GB is high; "
            "training will need more memory for optimizer states and gradients"
        )

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    parser.add_argument("--output-dir", default="./phase0_results")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--test-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[phase0] hidden state extraction")

    if not torch.cuda.is_available():
        print("warning: CUDA not available; VRAM measurements will be invalid")

    print(f"[load] model={args.model}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    except Exception as e:
        print(f"error: failed to load model: {e}")
        sys.exit(2)

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
        print(f"      allocated: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")

    # --- Identify attention layers ---
    print("[find] attention layers")
    attention_layers = find_attention_layers(model)
    if not attention_layers:
        print("result: fail (no attention layers identified)")
        report = {
            "extraction_possible": False,
            "error": "no attention layers identified",
        }
        with open(output_dir / "hidden_state_extraction.json", "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    for path, module in attention_layers[:10]:
        print(f"  {path} ({type(module).__name__})")
    if len(attention_layers) > 10:
        print(f"  ... and {len(attention_layers) - 10} more")

    # --- Run extraction ---
    print("[extract] running generation with hooks")
    extracted, vram_peak, metadata = extract_hidden_states(
        model, tokenizer, attention_layers,
        args.test_prompt, args.max_new_tokens,
    )

    # --- Analyze ---
    analysis = analyze_extraction(
        extracted, attention_layers, metadata, vram_peak
    )

    print(f"  layers hooked: {analysis['total_layers_hooked']}")
    print(f"  layers extracted: {analysis['layers_extracted']}")
    print(f"  layers failed: {analysis['layers_failed']}")
    print(f"  peak VRAM: {analysis['vram_peak_gb']:.1f} GB")
    if analysis["recommended_layers"]:
        print("  recommended layers:")
        for path in analysis["recommended_layers"]:
            info = analysis["per_layer"][path]
            print(f"    {path}  hidden_size={info.get('hidden_size')}  calls={info.get('num_calls')}")
    for warning in analysis["warnings"]:
        print(f"  warning: {warning}")

    # --- Write report ---
    report = {
        "model_id": args.model,
        "test_prompt": args.test_prompt,
        "max_new_tokens": args.max_new_tokens,
        **analysis,
        "metadata": metadata,
    }
    with open(output_dir / "hidden_state_extraction.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {output_dir / 'hidden_state_extraction.json'}")

    # --- Result ---
    if analysis["extraction_possible"]:
        print("result: pass")
        sys.exit(0)
    else:
        print("result: fail (extraction not possible)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
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
                }, _f, indent=2)
            print(f"crash report written to {_out / 'crash_report.json'}")
        except Exception:
            pass
        sys.exit(2)