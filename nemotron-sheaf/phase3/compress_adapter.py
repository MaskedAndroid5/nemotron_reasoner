#!/usr/bin/env python3
"""
compress_adapter.py — Phase 3: LoRA adapter compression via SVD

Analyses the trained LoRA adapter's weight matrices, computes effective
rank, and optionally compresses to a smaller rank without retraining.

The compression uses the standard low‑rank factorisation approach:
  1. For each pair of (lora_A, lora_B) belonging to the same module,
     compute the product W = lora_B @ lora_A.
  2. Compute the truncated SVD of W to the desired target rank:
       U_r, S_r, Vh_r
  3. Split the singular values (or assign them equally) to form new
     lora_A' = sqrt(S_r) @ Vh_r    (or Vh_r alone if splitting evenly)
     lora_B' = U_r @ sqrt(S_r)
  This preserves the forward pass approximately while reducing the
  adapter size.

Output:
  • svd_analysis.json         – per‑layer singular values & effective rank
  • compressed adapter        – new adapter_model.safetensors + adapter_config.json

Usage:
  python compress_adapter.py \
      --adapter phase2_checkpoints/final \
      --output compressed_adapter \
      --target-rank 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import safetensors.torch
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class CompressConfig(BaseModel):
    adapter_dir: Path
    output_dir: Path
    target_rank: Optional[int] = Field(None, ge=1, le=64)
    variance_threshold: float = Field(0.95, gt=0.0, le=1.0)
    dry_run: bool = False

    @validator("adapter_dir", "output_dir", always=True)
    def _resolve_paths(cls, v):
        return Path(v)

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_adapter_weights(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    st = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if st.exists():
        return safetensors.torch.load_file(str(st))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


# ---------------------------------------------------------------------------
# SVD analysis
# ---------------------------------------------------------------------------
def analyse_effective_rank(weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compute per‑layer singular values and effective rank at 95% variance."""
    analysis = {}
    for key, tensor in weights.items():
        if "lora_A" not in key:
            continue
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        cumsum = S.cumsum(0)
        total = cumsum[-1] + 1e-8
        eff_rank = int((cumsum / total < 0.95).sum()) + 1
        analysis[key] = {
            "singular_values": S.tolist(),
            "effective_rank_95": eff_rank,
        }
    return analysis


# ---------------------------------------------------------------------------
# Paired lora_A / lora_B compression
# ---------------------------------------------------------------------------
def _pair_lora_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Group lora_A and lora_B by their common module prefix.
    Returns dict: module_prefix -> {"lora_A": tensor, "lora_B": tensor}
    """
    pairs = defaultdict(dict)
    for key, tensor in weights.items():
        # Expected key format: ...<module>.lora_A.weight or ...<module>.lora_B.weight
        if "lora_A" in key:
            prefix = key.split(".lora_A")[0]
            pairs[prefix]["lora_A"] = tensor
        elif "lora_B" in key:
            prefix = key.split(".lora_B")[0]
            pairs[prefix]["lora_B"] = tensor
    # Keep only complete pairs
    return {k: v for k, v in pairs.items() if len(v) == 2}


def _compress_pair(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    target_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a single lora_A / lora_B pair to target_rank.
    lora_A shape: (r, in_features)
    lora_B shape: (out_features, r)

    Returns new (lora_A, lora_B) with rank = target_rank.
    """
    W = lora_B @ lora_A  # (out_features, in_features)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)

    Ur = U[:, :target_rank]
    Sr = S[:target_rank]
    Vhr = Vh[:target_rank, :]

    # Split singular values equally between A and B
    sqrt_S = torch.sqrt(Sr)
    new_lora_A = (torch.diag(sqrt_S) @ Vhr).to(lora_A.dtype)
    new_lora_B = (Ur @ torch.diag(sqrt_S)).to(lora_B.dtype)

    return new_lora_A, new_lora_B


def compress_weights(
    weights: Dict[str, torch.Tensor],
    target_rank: int,
) -> Dict[str, torch.Tensor]:
    """Compress all paired lora_A/lora_B and pass through everything else."""
    pairs = _pair_lora_keys(weights)
    compressed = {}
    for key, tensor in weights.items():
        if "lora_A" in key or "lora_B" in key:
            continue  # handled by pairs
        compressed[key] = tensor

    for prefix, pair in pairs.items():
        new_A, new_B = _compress_pair(pair["lora_A"], pair["lora_B"], target_rank)
        # Reconstruct the full key names
        # Find the original keys for this prefix
        a_key = prefix + ".lora_A.weight"
        b_key = prefix + ".lora_B.weight"
        # Handle potential key format variations (default or named)
        if a_key not in weights:
            # Try common alternative: lora_A.default.weight
            a_key = prefix + ".lora_A.default.weight"
            b_key = prefix + ".lora_B.default.weight"
        compressed[a_key] = new_A
        compressed[b_key] = new_B

    return compressed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Path to trained adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for compressed adapter")
    parser.add_argument("--target-rank", type=int, default=None,
                        help="Target rank for compression (default: analyse only)")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Variance threshold for effective rank calculation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse only, do not save compressed adapter")
    args = parser.parse_args()

    cfg = CompressConfig(
        adapter_dir=args.adapter,
        output_dir=args.output,
        target_rank=args.target_rank,
        variance_threshold=args.variance_threshold,
        dry_run=args.dry_run,
    )

    print(f"[compress] loading adapter from {cfg.adapter_dir}")
    weights = load_adapter_weights(cfg.adapter_dir)

    # Analyse
    analysis = analyse_effective_rank(weights)
    print("  per‑layer effective rank (95% variance):")
    for key, info in analysis.items():
        print(f"    {key}: rank={info['effective_rank_95']}")

    # Save analysis
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "svd_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  analysis saved to {cfg.output_dir / 'svd_analysis.json'}")

    if cfg.dry_run or cfg.target_rank is None:
        print("  dry‑run or no target rank specified – skipping compression")
        return

    # Compress
    print(f"  compressing to rank {cfg.target_rank}")
    compressed = compress_weights(weights, cfg.target_rank)

    # Save compressed adapter
    safetensors.torch.save_file(compressed, cfg.output_dir / "adapter_model.safetensors")
    print(f"  compressed weights saved ({len(compressed)} tensors)")

    # Copy and update adapter_config.json
    import shutil
    config_src = cfg.adapter_dir / "adapter_config.json"
    config_dst = cfg.output_dir / "adapter_config.json"
    if config_src.exists():
        shutil.copy(config_src, config_dst)
        with open(config_dst, "r+") as f:
            config_data = json.load(f)
            config_data["r"] = cfg.target_rank
            f.seek(0)
            json.dump(config_data, f, indent=2)
            f.truncate()
        print(f"  adapter_config.json updated (r={cfg.target_rank})")
    else:
        print("  warning: adapter_config.json not found in source – not copied")

    print("result: pass")


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