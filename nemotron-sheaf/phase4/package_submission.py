#!/usr/bin/env python3
"""
package_submission.py — Phase 4: submission packaging

Creates a competition‑ready submission.zip from a trained LoRA adapter
directory.  Validates that the required files exist and that the adapter
config contains a rank ≤ 32 before packaging.

Usage:
  python package_submission.py \
      --adapter phase2_checkpoints/final \
      --output submission/submission.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_adapter(adapter_dir: Path) -> None:
    """Check that the adapter directory is submission‑ready."""
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    if not weights_path.exists() and not bin_path.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors or adapter_model.bin not found in {adapter_dir}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    rank = config.get("r")
    if rank is None:
        raise ValueError("adapter_config.json is missing 'r' (rank)")
    if rank > 32:
        raise ValueError(
            f"LoRA rank is {rank}, but competition requires rank ≤ 32. "
            "Compress the adapter with compress_adapter.py first."
        )

    print(f"  adapter validated: rank={rank}, files present")


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def create_submission_zip(adapter_dir: Path, output_path: Path) -> None:
    """Zip the adapter files into a submission archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # adapter_config.json
        config_path = adapter_dir / "adapter_config.json"
        zf.write(config_path, arcname="adapter_config.json")

        # adapter_model.safetensors or .bin
        weights_path = adapter_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            weights_path = adapter_dir / "adapter_model.bin"
        zf.write(weights_path, arcname=weights_path.name)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  submission.zip created: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Path to trained adapter directory")
    parser.add_argument("--output", default="submission/submission.zip", help="Output zip path")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    output_path = Path(args.output)

    print("[package] submission packaging")
    print(f"  adapter: {adapter_dir}")
    print(f"  output: {output_path}")

    validate_adapter(adapter_dir)
    create_submission_zip(adapter_dir, output_path)
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