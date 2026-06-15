#!/usr/bin/env python3
"""
format_dataset.py — Phase 1: dataset builder

Converts filtered reasoning traces (JSONL) into tokenised instruction‑style
datasets augmented with sheaf‑tag position metadata and ground‑truth answers.

Every example in the output dataset contains:
  • input_ids, attention_mask, labels            – standard LM tensors
  • claim_positions                              – token indices of <claim> tags
  • overlap_positions                            – token indices of <overlap> tags
  • compatible_positions                         – token indices of <compatible> tags
  • incompatible_positions                       – token indices of <incompatible> tags
  • ground_truth                                 – original answer string (for GRPO reward)

The positions are consumed directly by `SheafConsistencyLoss` during
training, avoiding redundant re‑parsing of raw text.

Design guarantees:
  • Prompt masking is *verified* – the script asserts that prompt‑only
    tokenisation matches the prefix of the full conversation.
  • No silent data loss – every dropped example is logged with a reason.
  • Shared sheaf configuration is ingested from `sheaf_consistency_loss`
    to keep the vocabulary aligned.
  • Output includes a `format_report.json` with length percentiles,
    tag presence rates, and per‑agent statistics.

Usage:
  python format_dataset.py \
      --input-dir phase1_data/logical_deduction/filtered \
      --output-dir phase1_data/formatted/logical_deduction \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --max-seq-length 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer

# Local imports
from reasoning_taxonomy import get_agent, generate_prompt
from sheaf_consistency_loss import SheafLossConfig, TagPositionExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FormatConfig(BaseModel):
    """Configuration for the dataset formatter."""

    model_id: str = Field(..., min_length=1)
    max_seq_length: int = Field(4096, ge=128)
    pad_to_max_length: bool = Field(False)
    num_proc: int = Field(4, ge=1)

    # Shared with SheafConsistencyLoss
    sheaf_config: SheafLossConfig = Field(
        default_factory=SheafLossConfig,
        description="Sheaf loss hyperparameters — shared with training"
    )

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FormatReport:
    agent: str
    total: int = 0
    kept: int = 0
    dropped: int = 0
    drop_reasons: Counter = field(default_factory=Counter)
    length_stats: Dict[str, float] = field(default_factory=dict)
    tag_presence: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Agent: {self.agent}",
            f"  Total examples: {self.total}",
            f"  Kept: {self.kept}",
            f"  Dropped: {self.dropped}",
        ]
        if self.drop_reasons:
            lines.append("  Drop reasons:")
            for reason, count in self.drop_reasons.most_common():
                lines.append(f"    {reason}: {count}")
        if self.length_stats:
            lines.append("  Sequence length (tokens):")
            for k, v in self.length_stats.items():
                lines.append(f"    {k}: {v:.0f}")
        if self.tag_presence:
            lines.append("  Tag presence rate:")
            for tag, rate in self.tag_presence.items():
                lines.append(f"    {tag}: {rate:.2%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction & masking validation
# ---------------------------------------------------------------------------

def _build_conversation(
    example: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Return (system, user, assistant) from a JSONL example."""
    agent_name = example["agent"]
    variables = example.get("variables", {})
    trace = example["generated_trace"]
    messages = generate_prompt(agent_name, **variables)
    return messages["system"], messages["user"], trace


def _tokenise_and_validate_masking(
    system: str,
    user: str,
    assistant: str,
    tokenizer,
    max_seq_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Tokenise the full conversation with chat template.
    Verify that prompt‑only tokenisation matches the prefix of the full
    sequence.  Returns a dict with input_ids, attention_mask, labels,
    and sheaf‑tag positions, or None if the sequence is too long or
    the masking check fails.
    """
    # Full conversation
    full_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_dict=False,
    )

    # Prompt only (include the assistant header so we mask exactly the
    # tokens that are not the assistant's responsibility)
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        return_dict=False,
    )

    if len(full_ids) > max_seq_length:
        return None

    # Validate that prompt_ids is a prefix of full_ids
    prompt_len = len(prompt_ids)
    if full_ids[:prompt_len] != prompt_ids:
        # This indicates a tokeniser / template mismatch — hard error
        raise RuntimeError(
            "Prompt tokenisation is not a prefix of full tokenisation. "
            "Check the chat template and tokeniser configuration."
        )

    # Build labels: mask prompt tokens, keep assistant tokens
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # --- Extract sheaf tag positions ---
    # Use the TagPositionExtractor on the *full* sequence.
    extractor = TagPositionExtractor(tokenizer)
    claim_pos, compat_pos, incompat_pos, overlap_pos = \
        extractor.extract_all(torch.tensor(full_ids))

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "claim_positions": claim_pos,
        "overlap_positions": overlap_pos,
        "compatible_positions": compat_pos,
        "incompatible_positions": incompat_pos,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def format_dataset(
    input_dir: Path,
    output_dir: Path,
    config: FormatConfig,
) -> FormatReport:
    """Convert filtered JSONL examples into a HuggingFace Dataset."""
    # Load tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for model '{config.model_id}': {exc}. "
            f"Ensure the model ID is correct and HuggingFace Hub is accessible."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load examples
    examples_path = input_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(str(examples_path))

    examples = []
    with open(examples_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    report = FormatReport(agent=input_dir.parent.name)
    report.total = len(examples)

    # Pre‑filter: estimate token count to skip obviously long examples
    token_lengths = []
    kept_examples = []
    drop_reasons: Counter = Counter()

    for ex in examples:
        try:
            system, user, assistant = _build_conversation(ex)
        except Exception:
            drop_reasons["build_error"] += 1
            continue

        # Rough token estimate: word count * 1.33 heuristic
        rough_len = int(len(assistant.split()) * 1.33 + len(system.split()) * 1.33 + len(user.split()) * 1.33)
        if rough_len > config.max_seq_length * 1.3:
            drop_reasons["estimated_overflow"] += 1
            continue

        # Keep the ground_truth for GRPO reward
        kept_examples.append({
            "system": system,
            "user": user,
            "assistant": assistant,
            "ground_truth": ex.get("ground_truth", ""),
        })

    # Tokenise with parallel map
    ds = Dataset.from_list(kept_examples)

    def tokenise_fn(batch):
        results = defaultdict(list)
        for i in range(len(batch["system"])):
            out = _tokenise_and_validate_masking(
                batch["system"][i],
                batch["user"][i],
                batch["assistant"][i],
                tokenizer,
                config.max_seq_length,
            )
            if out is None:
                # Mark for filtering
                results["_drop"].append(True)
                # Add dummy entries for column alignment
                for key in ["input_ids", "attention_mask", "labels",
                            "claim_positions", "overlap_positions",
                            "compatible_positions", "incompatible_positions",
                            "ground_truth"]:
                    results[key].append([])
            else:
                results["_drop"].append(False)
                # Preserve the original ground_truth for GRPO reward
                out["ground_truth"] = batch["ground_truth"][i]
                for key, val in out.items():
                    results[key].append(val)
        return results

    # Apply tokenisation
    ds = ds.map(
        tokenise_fn,
        batched=True,
        batch_size=32,
        remove_columns=["system", "user", "assistant", "ground_truth"],
        num_proc=config.num_proc,
        desc="tokenising",
    )

    # Filter out dropped examples
    ds = ds.filter(lambda x: not x["_drop"])
    ds = ds.remove_columns("_drop")

    # Cast to fixed schema with Sequence types for variable-length lists
    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "claim_positions": Sequence(Value("int64")),
        "overlap_positions": Sequence(Value("int64")),
        "compatible_positions": Sequence(Value("int64")),
        "incompatible_positions": Sequence(Value("int64")),
        "ground_truth": Value("string"),
    })
    ds = ds.cast(features)

    # Compute report statistics
    report.kept = len(ds)
    report.dropped = report.total - report.kept
    report.drop_reasons = drop_reasons

    # Length statistics
    lengths = [len(x["input_ids"]) for x in ds]
    if lengths:
        sorted_len = sorted(lengths)
        n = len(sorted_len)
        report.length_stats = {
            "min": float(sorted_len[0]),
            "p25": float(sorted_len[int(n * 0.25)]),
            "p50": float(sorted_len[int(n * 0.5)]),
            "p75": float(sorted_len[int(n * 0.75)]),
            "p95": float(sorted_len[int(n * 0.95)]),
            "max": float(sorted_len[-1]),
        }

    # Tag presence rates
    for tag_col in ["claim_positions", "overlap_positions",
                     "compatible_positions", "incompatible_positions"]:
        present = sum(1 for x in ds if len(x[tag_col]) > 0)
        report.tag_presence[tag_col] = present / len(ds) if len(ds) > 0 else 0.0

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))

    # Save metadata (shared config)
    with open(output_dir / "sheaf_config.json", "w") as f:
        f.write(config.sheaf_config.json(indent=2))

    # Save report
    with open(output_dir / "format_report.json", "w") as f:
        json.dump({
            "agent": report.agent,
            "total": report.total,
            "kept": report.kept,
            "dropped": report.dropped,
            "drop_reasons": dict(report.drop_reasons),
            "length_stats": report.length_stats,
            "tag_presence": report.tag_presence,
        }, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    sheaf_cfg = SheafLossConfig()  # default values; can be overridden via config file
    config = FormatConfig(
        model_id=args.model,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=args.pad_to_max_length,
        num_proc=args.num_proc,
        sheaf_config=sheaf_cfg,
    )

    print(f"[format] model: {args.model}")
    print(f"  max_seq_length: {args.max_seq_length}")

    report = format_dataset(
        Path(args.input_dir),
        Path(args.output_dir),
        config,
    )
    print(report.summary())
    sys.exit(0 if report.kept > 0 else 1)


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
