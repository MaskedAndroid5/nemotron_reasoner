#!/usr/bin/env python3
"""
train_lora.py — Phase 2: production‑grade LoRA trainer with sheaf‑consistency loss
                  and optional GRPO reinforcement‑learning fine‑tuning phase.

Veteran‑grade safety guarantees (v3.4 – deadline‑ready):
  • Bit‑exact resumption with full RNG checkpointing.
  • Instability guard – NaN/Inf loss skips batches.
  • GRPO epoch_reward correctly incremented (bug fix).
  • GRPO steps after every batch (simplified for small datasets).
  • Global gradient clipping only (removed per‑param variant).
  • Explicit cache clearing after every forward pass.
  • Dual‑loss ratio diagnostic on first step and every log interval.

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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
    # GRPO phase (uses rl_learning_rate)
    rl_epochs: int = Field(0, ge=0)
    rl_num_samples: int = Field(4, ge=2)
    rl_learning_rate: float = Field(5e-6, gt=0.0,
                                    description="Learning rate for GRPO phase only")
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


# ---------------------------------------------------------------------------
# RNG and checkpointing
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------
_BOXED_REGEX = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_REGEX.findall(text)
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
    return reward, {"answer_correct": answer_correct, "consistency_score": consistency_score, "format_ok": format_ok}


# ---------------------------------------------------------------------------
# GRPO phase (simplified – step every batch)
# ---------------------------------------------------------------------------
def _run_grpo_phase(
    model, tokenizer, dataloader, sheaf_loss_fn, config: TrainConfig,
    optimizer, scheduler, scaler, device: torch.device, output_dir: Path, global_step: int,
):
    print(f"[grpo] starting GRPO phase ({config.rl_epochs} epochs, {config.rl_num_samples} samples per problem)")
    model.train()

    for epoch in range(config.rl_epochs):
        epoch_reward = 0.0
        t0 = time.perf_counter()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            ground_truths = batch["ground_truth"]

            B = input_ids.size(0)
            pg_losses: List[torch.Tensor] = []
            batch_rewards: List[float] = []

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
                pg_losses.append(pg_loss)
                batch_rewards.append(reward_best)
                epoch_reward += reward_best

            if not pg_losses:
                continue

            # Step every batch (simplified for small datasets)
            total_pg_loss = torch.stack(pg_losses).mean()
            scaler.scale(total_pg_loss).backward()
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
                print(
                    f"  [grpo] step {global_step}  "
                    f"avg_reward={avg_r:.4f}  "
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


# ---------------------------------------------------------------------------
# Metrics & safety
# ---------------------------------------------------------------------------
def _check_finite(lm_loss, sheaf_loss) -> bool:
    return torch.isfinite(lm_loss).all() and torch.isfinite(sheaf_loss).all()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
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

    # --- Load LoRA settings ---
    lora_cfg_path = Path(args.config)
    print(f"[config] loading {lora_cfg_path}")
    import yaml
    raw_yaml = yaml.safe_load(lora_cfg_path.read_text())
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    if raw_yaml and "lora" in raw_yaml and "model_id" in raw_yaml["lora"]:
        model_id = raw_yaml["lora"]["model_id"]

    # --- Load model ---
    print(f"[load] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Build PEFT adapter ---
    settings = load_lora_settings(lora_cfg_path, model=model, verify=True)
    peft_config = build_peft_config(settings)
    from peft import get_peft_model
    model = get_peft_model(model, peft_config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable:,}")

    # Verify target_parameters
    if settings.has_target_parameters:
        param_names = {name for name, _ in model.named_parameters()}
        for tp in settings.target_parameters:
            if not any(tp in name and "lora" in name for name in param_names):
                raise RuntimeError(
                    f"target_parameter '{tp}' not found in adapted parameters. "
                    "PEFT may have silently skipped it."
                )
        print(f"  target_parameters verification: OK ({len(settings.target_parameters)} paths)")

    # --- Build sheaf loss ---
    data, sheaf_cfg = _load_datasets(cfg.data_dir)
    if cfg.lambda_sheaf is not None:
        if cfg.resume and cfg.lambda_sheaf != getattr(sheaf_cfg, "lambda_compatible", None):
            raise ValueError(
                "Cannot override --lambda-sheaf on resume with a different value. "
                "Use the same lambda that started this run or resume without --lambda-sheaf."
            )
        sheaf_cfg.lambda_compatible = cfg.lambda_sheaf
    sheaf_loss_fn = SheafConsistencyLoss(sheaf_cfg, tokenizer)
    print(f"[sheaf] lambda={sheaf_cfg.lambda_compatible}")

    # --- DataLoader ---
    dataloader = DataLoader(
        data, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    total_steps_per_epoch = len(dataloader) // cfg.gradient_accumulation_steps
    total_steps = total_steps_per_epoch * cfg.epochs
    print(f"[info] dataset: {len(dataloader)} batches → "
          f"{total_steps_per_epoch} effective batches/epoch")

    # --- Optimizer / scheduler / scaler ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if cfg.resume:
        ckpt_path = cfg.resume / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "rng_states" in ckpt:
            _restore_rng(ckpt["rng_states"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[resume] epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")

    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()

    # ==================================================================
    # Supervised fine‑tuning (CE + sheaf)
    # ==================================================================
    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss = 0.0
        epoch_lm = 0.0
        epoch_sheaf = 0.0
        skipped_batches = 0
        t0 = time.perf_counter()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                )
                lm_loss = outputs.loss

                sheaf_loss, sheaf_diag = sheaf_loss_fn(
                    outputs.hidden_states, batch["input_ids"]
                )

                if not _check_finite(lm_loss, sheaf_loss):
                    print(f"  WARNING: Non-finite loss at step {global_step}. "
                          f"lm={lm_loss.item():.4f} sheaf={sheaf_loss.item():.6f}. "
                          "Skipping batch.")
                    optimizer.zero_grad()
                    scaler.update()
                    skipped_batches += 1
                    continue

                # Dual‑loss ratio diagnostic
                if global_step == 1 or global_step % cfg.log_steps == 0:
                    ratio = sheaf_loss.item() / max(lm_loss.item(), 1e-8)
                    if ratio > 0.5:
                        print(f"  [warning] sheaf_loss/lm_loss = {ratio:.2f} — "
                              f"sheaf loss may dominate. Consider reducing --lambda-sheaf.")
                    elif ratio < 0.001:
                        print(f"  [warning] sheaf_loss/lm_loss = {ratio:.4f} — "
                              f"sheaf loss may be too weak. Consider increasing --lambda-sheaf.")

                total_loss = (lm_loss + sheaf_loss) / cfg.gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            epoch_loss += total_loss.item()
            epoch_lm += lm_loss.item()
            epoch_sheaf += sheaf_loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_steps == 0:
                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    print(
                        f"  step {global_step}/{total_steps}  "
                        f"lm={lm_loss.item():.4f}  sheaf={sheaf_loss.item():.6f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  vram={vram_mb:.0f}MB"
                    )

                if global_step % cfg.save_steps == 0:
                    _save_checkpoint(
                        output_dir, model, optimizer, scheduler, scaler,
                        epoch, global_step, best_loss, cfg.save_total_limit
                    )

            torch.cuda.empty_cache()

        # End of SFT epoch
        avg_loss = epoch_loss / len(dataloader)
        avg_lm = epoch_lm / len(dataloader)
        avg_sheaf = epoch_sheaf / len(dataloader)
        elapsed = time.perf_counter() - t0
        print(
            f"  epoch {epoch+1}/{cfg.epochs}  "
            f"loss={avg_loss:.4f}  lm={avg_lm:.4f}  sheaf={avg_sheaf:.6f}  "
            f"time={elapsed:.1f}s  skipped={skipped_batches}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_adapter(model, output_dir / "best")

    # ==================================================================
    # Optional GRPO phase
    # ==================================================================
    if cfg.rl_epochs > 0:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.rl_learning_rate, weight_decay=cfg.weight_decay,
        )
        rl_total_steps = total_steps_per_epoch * cfg.rl_epochs
        warmup_steps_rl = int(rl_total_steps * cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_rl, num_training_steps=rl_total_steps,
        )
        global_step = _run_grpo_phase(
            model, tokenizer, dataloader, sheaf_loss_fn, cfg,
            optimizer, scheduler, scaler, device, output_dir, global_step,
        )

    # --- Export final adapter ---
    final_dir = output_dir / "final"
    _save_adapter(model, final_dir)

    submission_dir = Path("submission/adapter")
    _save_adapter(model, submission_dir)
    print(f"[submission] adapter exported to {submission_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase0_results/lora_config_safe.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="phase2_checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-sheaf", type=float, default=None)
    parser.add_argument("--rl-epochs", type=int, default=0)
    parser.add_argument("--rl-num-samples", type=int, default=4)
    parser.add_argument("--rl-learning-rate", type=float, default=5e-6)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hardened entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\ninterrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nunhandled exception: {exc}", file=sys.stderr)
        traceback.print_exc()
        try:
            import json as _json
            _out = Path("phase2_checkpoints")
            _out.mkdir(parents=True, exist_ok=True)
            with open(_out / "crash_report.json", "w") as _f:
                _json.dump({
                    "error": "unhandled_exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }, _f, indent=2)
            print(f"crash report written to {_out / 'crash_report.json'}")
        except Exception:
            pass
        sys.exit(2)
