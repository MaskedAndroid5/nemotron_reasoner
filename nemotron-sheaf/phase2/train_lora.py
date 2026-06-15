#!/usr/bin/env python3
"""
train_lora.py — Phase 2: LoRA trainer with sheaf‑consistency loss
                  and optional GRPO reinforcement‑learning fine‑tuning phase.

Safety guarantees (v3.2):
  • Bit‑exact resumption with full RNG checkpointing
  • Instability guard — NaN/Inf loss skips batches
  • GRPO memory hardening — explicit cache clearing after generation
  • Dual‑loss ratio diagnostics — warns if sheaf dominates or vanishes
  • Per‑parameter gradient clipping
  • Config immutability on resume

Usage:
  python train_lora.py \
      --config phase0_results/lora_config_safe.yaml \
      --data-dir phase1_data/formatted \
      --output-dir phase2_checkpoints \
      --epochs 3 --batch-size 1 --gradient-accumulation 4 \
      --learning-rate 1e-4 --rl-epochs 1 --rl-num-samples 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, validator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk, concatenate_datasets

from lora_config_loader import load_lora_settings, build_peft_config, LoRASettings
from sheaf_consistency_loss import SheafConsistencyLoss, SheafLossConfig


class TrainConfig(BaseModel):
    data_dir: Path
    max_seq_length: int = Field(4096, ge=128)
    epochs: int = Field(3, ge=1)
    batch_size: int = Field(1, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(1e-4, gt=0.0)
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    use_amp: bool = Field(True)
    lambda_sheaf: Optional[float] = Field(None, ge=0.0)
    rl_epochs: int = Field(0, ge=0)
    rl_num_samples: int = Field(4, ge=2)
    rl_learning_rate: float = Field(5e-6, gt=0.0)
    rl_temperature: float = Field(0.8, gt=0.0, le=1.0)
    output_dir: Path
    log_steps: int = Field(10, ge=1)
    save_steps: int = Field(200, ge=1)
    save_total_limit: int = Field(3, ge=1)
    resume: Optional[Path] = Field(None)
    seed: int = Field(42, ge=0)

    @validator("output_dir", always=True)
    def _resolve_output(cls, v):
        return Path(v)

    class Config:
        extra = "forbid"


@dataclass
class TrainerState:
    epoch: int
    global_step: int
    best_loss: float
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    scaler_state_dict: Optional[Dict[str, Any]] = None
    rng_states: Dict[str, Any] = field(default_factory=dict)


def _capture_rng() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: Dict[str, Any]):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    ground_truths = []
    for i, ex in enumerate(batch):
        seq_len = len(ex["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = torch.tensor(ex["labels"], dtype=torch.long)
        ground_truths.append(ex.get("ground_truth", ""))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ground_truth": ground_truths,
        "claim_positions": [ex["claim_positions"] for ex in batch],
        "overlap_positions": [ex["overlap_positions"] for ex in batch],
        "compatible_positions": [ex["compatible_positions"] for ex in batch],
        "incompatible_positions": [ex["incompatible_positions"] for ex in batch],
    }


def _load_datasets(data_dir: Path) -> Tuple[Dataset, SheafLossConfig]:
    parts = []
    sheaf_cfg = SheafLossConfig()
    for agent_dir in sorted(data_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        ds_path = agent_dir
        if not (ds_path / "dataset_info.json").exists():
            continue
        ds = load_from_disk(str(ds_path))
        parts.append(ds)
        sc_path = agent_dir / "sheaf_config.json"
        if sc_path.exists():
            sheaf_cfg = SheafLossConfig.parse_file(sc_path)
    if not parts:
        raise FileNotFoundError(f"No formatted datasets found in {data_dir}")
    combined = concatenate_datasets(parts)
    print(f"  combined dataset: {len(combined)} examples")
    return combined, sheaf_cfg


def _extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', text)
    return matches[-1].strip() if matches else None


def _compute_sheaf_consistency_reward(
    trace: str, model, tokenizer, sheaf_loss_fn: SheafConsistencyLoss, device: torch.device
) -> float:
    try:
        inputs = tokenizer(trace, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        loss, _ = sheaf_loss_fn(outputs.hidden_states, inputs["input_ids"])
        del outputs
        torch.cuda.empty_cache()
        reward = 1.0 / (1.0 + loss.item())
        return float(reward)
    except Exception as exc:
        print(f"  [warning] sheaf reward computation failed: {exc}")
        return 0.0


def _compute_grpo_reward(
    trace: str, ground_truth: str, model, tokenizer,
    sheaf_loss_fn: SheafConsistencyLoss, device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    extracted = _extract_boxed(trace)
    answer_correct = 1.0 if (extracted is not None and extracted.strip().lower() == ground_truth.strip().lower()) else 0.0
    consistency_score = _compute_sheaf_consistency_reward(trace, model, tokenizer, sheaf_loss_fn, device)
    has_boxed = "\\boxed{" in trace
    format_ok = 1.0 if has_boxed else 0.0
    reward = 0.5 * answer_correct + 0.3 * consistency_score + 0.2 * format_ok
    diagnostics = {
        "answer_correct": answer_correct,
        "consistency_score": consistency_score,
        "format_ok": format_ok,
    }
    return reward, diagnostics


def _run_grpo_phase(
    model, tokenizer, dataloader, sheaf_loss_fn, config: TrainConfig,
    optimizer, scheduler, scaler, device: torch.device, output_dir: Path, global_step: int,
):
    print(f"[grpo] starting GRPO phase ({config.rl_epochs} epochs, {config.rl_num_samples} samples per problem)")
    model.train()

    for epoch in range(config.rl_epochs):
        epoch_reward = 0.0
        t0 = time.perf_counter()
        batch_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            ground_truths = batch["ground_truth"]

            B = input_ids.size(0)
            batch_rewards: List[float] = []
            batch_advantages: List[float] = []

            for i in range(B):
                prompt_mask = labels[i] == -100
                prompt_tokens = input_ids[i][prompt_mask]
                prompt_len = prompt_tokens.size(0)
                if prompt_len == 0:
                    continue

                gt = ground_truths[i] if i < len(ground_truths) else ""
                traces = []
                rewards = []
                for _ in range(config.rl_num_samples):
                    with torch.no_grad():
                        gen_out = model.generate(
                            input_ids=prompt_tokens.unsqueeze(0),
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=config.rl_temperature,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    gen_ids = gen_out[0, prompt_len:]
                    trace_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    del gen_out
                    torch.cuda.empty_cache()
                    reward, _ = _compute_grpo_reward(
                        trace_text, gt, model, tokenizer, sheaf_loss_fn, device
                    )
                    traces.append((prompt_tokens, gen_ids, trace_text, reward))
                    rewards.append(reward)

                if not rewards:
                    continue

                mean_reward = sum(rewards) / len(rewards)
                advantages = [r - mean_reward for r in rewards]

                best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
                if advantages[best_idx] <= 0:
                    continue

                prompt_tokens_best, gen_ids_best, trace_text_best, reward_best = traces[best_idx]
                advantage_best = advantages[best_idx]

                full_ids = torch.cat([prompt_tokens_best, gen_ids_best], dim=0).unsqueeze(0)
                full_labels = full_ids.clone()
                full_labels[:, :prompt_len] = -100

                outputs = model(input_ids=full_ids, labels=full_labels)
                log_prob = -outputs.loss

                pg_loss = -advantage_best * log_prob
                scaler.scale(pg_loss).backward()

                epoch_reward += reward_best
                batch_rewards.append(reward_best)
                batch_advantages.append(advantage_best)

            batch_count += 1
            if batch_count % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_steps == 0:
                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    avg_r = sum(batch_rewards) / max(1, len(batch_rewards))
                    avg_a = sum(batch_advantages) / max(1, len(batch_advantages))
                    print(
                        f"  [grpo] step {global_step}  "
                        f"avg_reward={avg_r:.4f}  avg_adv={avg_a:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  vram={vram_mb:.0f}MB"
                    )
                if global_step % config.save_steps == 0:
                    _save_checkpoint(
                        output_dir, model, optimizer, scheduler, scaler,
                        epoch + config.epochs, global_step, float("inf"),
                        config.save_total_limit
                    )

        avg_reward = epoch_reward / len(dataloader) if len(dataloader) > 0 else 0
        elapsed = time.perf_counter() - t0
        print(f"  [grpo] epoch {epoch+1}/{config.rl_epochs}  avg_reward={avg_reward:.4f}  time={elapsed:.1f}s")

    print("[grpo] GRPO phase complete")
    return global_step


def _compute_grad_norms(model) -> Dict[str, float]:
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "lora" in name and "target_parameter" in name:
            group = "lora_moe"
        elif "lora" in name:
            group = "lora_attn"
        else:
            group = "other"
        key = f"grad/{group}"
        norms[key] = norms.get(key, 0.0) + param.grad.norm().item() ** 2
    return {k: v ** 0.5 for k, v in norms.items()}


def _check_finite(lm_loss, sheaf_loss) -> bool:
    return torch.isfinite(lm_loss).all() and torch.isfinite(sheaf_loss).all()


def _clip_grad_per_param(model, max_norm):
    for param in model.parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_([param], max_norm)


def _save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                     epoch, global_step, best_loss, limit):
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_states": _capture_rng(),
    }, ckpt_dir / "checkpoint.pt")
    existing = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in existing[:-limit]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    print(f"  [save] {ckpt_dir}")


def _save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    print(f"  [adapter] {path}")


def train():
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        resume=args.resume,
        lambda_sheaf=args.lambda_sheaf,
        rl_epochs=args.rl_epochs,
        rl_num_samples=args.rl_num_samples,
        rl_learning_rate=args.rl_learning_rate,
    )
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("[train] production‑grade LoRA trainer (veteran + GRPO)")
    print(f"  config: {cfg.json(indent=2)}")

    lora_cfg_path = Path(args.config)
    print(f"[config] loading {lora_cfg_path}")
    import yaml
    raw_yaml = yaml.safe_load(lora_cfg_path.read_text())
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    if raw_yaml and "lora" in raw_yaml and "model_id" in raw_yaml["lora"]:
        model_id = raw_yaml["lora"]["model_id"]

    print(f"[load] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    settings = load_lora_settings(lora_cfg_path, model=model, verify=True)
    peft_config = build_peft_config(settings)
    from peft import get_peft_model
    model = get_peft_model(model, peft_config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable:,}")

    if settings.has_target_parameters:
        param_names = {name for name, _ in model.named_parameters()}
        for tp in settings.target_parameters:
            if not any(tp in name and "lora" in name for name in param_names):
                raise RuntimeError(
                    f"target_parameter '{tp}' not found in adapted parameters. "
                    "PEFT may have silently skipped it."
                )
        print(f"  target_parameters verification: OK ({len(settings.target_parameters)} paths)")

    data, sheaf_cfg = _load_datasets(cfg.data_dir)
    if cfg.lambda_sheaf is not None:
        if cfg.resume and cfg.lambda
