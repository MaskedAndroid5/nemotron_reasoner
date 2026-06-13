#!/usr/bin/env python3
"""
consistency_loop.py — Phase 2: Multi‑Agent Consistency Loop (vLLM + HF hybrid)

Given a reasoning problem, spawns multiple reasoning attempts using the
agent router, cross‑checks their structured traces via the sheaf‑consistency
loss, and synthesises the most consistent final answer.

Critical fixes for Nemotron‑3‑Nano‑30B‑A3B‑BF16 compatibility:
  1. **Hidden state extraction** – vLLM does not expose hidden states.
     We use a separate HuggingFace `AutoModelForCausalLM` with the
     trained LoRA adapter (via PEFT) to extract token‑level hidden
     states on the generated traces.  This adds a small latency overhead
     but is necessary for sheaf consistency scoring.
  2. **Prompt template** – Uses `apply_chat_template` with the official
     Nemotron chat format (including `<think>` tags for reasoning).
  3. **LoRA validation** – Before scoring, verifies that the adapter
     contains sheaf projection weights.  If not, raises a clear error.
  4. **Tag extraction** – Uses token positions (via the tokenizer)
     instead of character offsets to match sheaf loss expectations.
  5. **Memory management** – The HuggingFace model can be offloaded to
     CPU (`--hf-device cpu`) to avoid OOM when running alongside vLLM.
     Using the FP8 model variant is recommended for single‑GPU setups.
  6. **Fallback scoring** – If hidden state extraction fails (e.g. OOM
     or adapter format mismatch), falls back to heuristic scoring.

Requirements
------------
  • A trained LoRA adapter with sheaf projections (saved by Phase 2).
  • vLLM ≥ 0.12.0 for generation.
  • HuggingFace `transformers` + PEFT for hidden state extraction.
  • GPU with ≥ 40 GB VRAM (or use FP8 variant with `--hf-device cpu`).
  • Set `HF_TOKEN` and `HF_HOME` environment variables.

Usage
-----
  python consistency_loop.py \
      --problem "If x + 3 = 7, what is x?" \
      --adapter phase2_checkpoints/final \
      --num-attempts 3 \
      --hf-device cpu \
      --output answer.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Local imports
from agent_router import AgentRouter, RouterConfig
from reasoning_taxonomy import get_agent, generate_prompt
from sheaf_consistency_loss import SheafConsistencyLoss, SheafLossConfig, TagPositionExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class LoopConfig(BaseModel):
    problem: str
    adapter_dir: Path
    model_id: str = Field("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    output_path: Optional[Path] = None

    # Spawning
    num_attempts: int = Field(3, ge=1, le=16)
    max_extra_attempts: int = Field(3, ge=0, le=8)
    temperature: float = Field(0.6, gt=0.0, le=1.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(1024, ge=64)

    # Consistency
    inconsistency_threshold: float = Field(0.5, gt=0.0, description="Max acceptable inconsistency score before spawning more attempts")
    sheaf_config: SheafLossConfig = Field(default_factory=SheafLossConfig)

    # vLLM
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=512)
    max_lora_rank: int = Field(32, ge=1)

    # HuggingFace for hidden states
    hf_device: str = Field("auto", description="Device for HF model: 'auto', 'cpu', or 'cuda:0'")

    # Router
    router_config: RouterConfig = Field(default_factory=RouterConfig)

    @validator("adapter_dir", "output_path", always=True)
    def _resolve_paths(cls, v):
        return Path(v) if v is not None else None

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ReasoningAttempt:
    agent_name: str
    trace: str
    extracted_answer: Optional[str]
    claim_positions: List[int]   # token positions
    overlap_positions: List[int]
    compatible_positions: List[int]
    incompatible_positions: List[int]
    inconsistency_score: float = float("inf")
    latency_ms: float = 0.0


@dataclass
class LoopResult:
    problem: str
    final_answer: str
    confidence: float
    attempts: List[ReasoningAttempt] = field(default_factory=list)
    selected_agent: str = ""
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------
_BOXED_REGEX = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_REGEX.findall(text)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------
class ConsistencyLoop:
    """Multi‑agent reasoning with sheaf‑based cross‑checking."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.router = AgentRouter(config.router_config)
        self._init_vllm()
        self._init_hf_model()
        self._init_sheaf_loss()

    def _init_vllm(self):
        print(f"[vllm] loading model {self.config.model_id}")
        self.llm = LLM(
            model=self.config.model_id,
            enable_lora=True,
            max_lora_rank=self.config.max_lora_rank,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )
        self.lora_req = LoRARequest(
            lora_name="consistency",
            lora_int_id=1,
            lora_path=str(self.config.adapter_dir),
        )

    def _init_hf_model(self):
        """Load the base model with LoRA adapter via HuggingFace for hidden states."""
        device = self.config.hf_device
        print(f"[hf] loading base model for hidden state extraction (device={device}) ...")
        # Warn if loading to GPU alongside vLLM
        if device != "cpu":
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram < 80:
                print(f"[hf] WARNING: GPU has {total_vram:.1f} GB. Loading BF16 model (~40 GB) "
                      "alongside vLLM may OOM. Use --hf-device cpu or FP8 variant.")
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            # Attach the LoRA adapter
            self.hf_model = PeftModel.from_pretrained(
                self.base_model, str(self.config.adapter_dir)
            )
            self.hf_model.eval()
        except (ValueError, RuntimeError) as e:
            if "peft" in str(e).lower() or "lora" in str(e).lower():
                raise RuntimeError(
                    "Failed to load adapter via PEFT. The adapter may have been trained "
                    "with NeMo's custom LoRA format. Please convert it to HuggingFace/PEFT "
                    "format, or set --hf-device cpu and use heuristic scoring fallback."
                ) from e
            raise
        print("[hf] model ready.")

    def _unload_hf_model(self):
        """Release HF model memory."""
        if hasattr(self, "hf_model"):
            del self.hf_model
            del self.base_model
            torch.cuda.empty_cache()
            print("[hf] model unloaded.")

    def _init_sheaf_loss(self):
        """Load sheaf consistency module and verify projection weights exist."""
        self.sheaf_loss = SheafConsistencyLoss(
            self.config.sheaf_config, self.tokenizer
        )
        # Load projection weights from the adapter
        st_path = self.config.adapter_dir / "adapter_model.safetensors"
        if not st_path.exists():
            raise FileNotFoundError(
                f"Adapter weights not found at {st_path}. "
                "Train the adapter with sheaf loss first."
            )

        from safetensors import safe_open
        with safe_open(str(st_path), framework="pt") as f:
            keys = list(f.keys())

        sheaf_keys = [k for k in keys if "sheaf_loss" in k or "projections" in k]
        if not sheaf_keys:
            raise RuntimeError(
                "The adapter does not contain sheaf projection weights. "
                "Ensure the adapter was trained with the sheaf consistency loss enabled."
            )

        state = {k: f.get_tensor(k) for k in sheaf_keys}
        cleaned_state = {}
        for k, v in state.items():
            new_k = k
            for prefix in ["sheaf_loss.", "projections."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            cleaned_state[new_k] = v

        missing, unexpected = self.sheaf_loss.load_state_dict(cleaned_state, strict=False)
        if missing:
            print(f"[sheaf] warning: missing keys in adapter: {missing}")
        if unexpected:
            print(f"[sheaf] warning: unexpected keys in adapter: {unexpected}")
        print("[sheaf] loaded projection weights from adapter")

    def _run_attempt(self, agent_name: str) -> ReasoningAttempt:
        """Generate a single reasoning trace using vLLM."""
        t0 = time.perf_counter()
        agent = get_agent(agent_name)
        messages = generate_prompt(agent_name, problem=self.config.problem)

        # Official chat template (includes <think> tags)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.llm.generate(
            [prompt], self.sampling_params, lora_request=self.lora_req
        )
        trace = outputs[0].outputs[0].text
        latency = (time.perf_counter() - t0) * 1000

        # Extract tag positions using tokenizer (token indices)
        agent_tags = list(agent.output_tags.keys())
        token_ids = self.tokenizer.encode(trace, add_special_tokens=False)

        # Create a temporary TagPositionExtractor to find positions
        extractor = TagPositionExtractor(self.tokenizer)
        claim_pos = extractor.find_positions(torch.tensor(token_ids), extractor.CLAIM_TAGS)
        overlap_pos = extractor.find_positions(torch.tensor(token_ids), extractor.OVERLAP_TAGS)
        compat_pos = extractor.find_positions(torch.tensor(token_ids), extractor.COMPATIBLE_TAGS)
        incompat_pos = extractor.find_positions(torch.tensor(token_ids), extractor.INCOMPATIBLE_TAGS)

        return ReasoningAttempt(
            agent_name=agent_name,
            trace=trace,
            extracted_answer=_extract_boxed(trace),
            claim_positions=claim_pos,
            overlap_positions=overlap_pos,
            compatible_positions=compat_pos,
            incompatible_positions=incompat_pos,
            latency_ms=latency,
        )

    def _compute_inconsistency(
        self, attempts: List[ReasoningAttempt]
    ) -> List[ReasoningAttempt]:
        """
        Score each attempt by running the sheaf loss on the trace using
        the HuggingFace model (which provides hidden states).
        Falls back to heuristic scoring if the HF model is unavailable or OOM.
        """
        device = next(self.hf_model.parameters()).device

        for attempt in attempts:
            if not attempt.claim_positions or not attempt.overlap_positions:
                attempt.inconsistency_score = float("inf")
                continue

            try:
                inputs = self.tokenizer(attempt.trace, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = self.hf_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (B, T, D)
                input_ids = inputs["input_ids"]

                # SheafLoss expects hidden_states tuple and input_ids tensor
                loss, _ = self.sheaf_loss(hidden_states, input_ids)
                attempt.inconsistency_score = loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    [OOM] Falling back to heuristic scoring.")
                    # Clear cache and use heuristic
                    torch.cuda.empty_cache()
                    score = 0.0
                    if not attempt.compatible_positions and not attempt.incompatible_positions:
                        score += 1.0
                    if len(attempt.claim_positions) < 2:
                        score += 1.0
                    attempt.inconsistency_score = score
                else:
                    raise
            except Exception as e:
                print(f"    warning: sheaf loss computation failed: {e}")
                score = 0.0
                if not attempt.compatible_positions and not attempt.incompatible_positions:
                    score += 1.0
                if len(attempt.claim_positions) < 2:
                    score += 1.0
                attempt.inconsistency_score = score

        return attempts

    def solve(self) -> LoopResult:
        """Run the full consistency loop and return the best answer."""
        t0 = time.perf_counter()

        # 1. Classify
        candidates = self.router.classify(self.config.problem)
        primary_agent = candidates[0][0] if candidates else "logical_deduction"
        print(f"[router] selected agent: {primary_agent}")

        # 2. Spawn initial attempts
        attempts: List[ReasoningAttempt] = []
        agents_to_try = [primary_agent] + [
            name for name, _ in candidates[1:] if name != primary_agent
        ]

        for i in range(self.config.num_attempts):
            agent = agents_to_try[i % len(agents_to_try)] if agents_to_try else primary_agent
            print(f"  [{i+1}/{self.config.num_attempts}] {agent} …", end=" ", flush=True)
            attempt = self._run_attempt(agent)
            attempts.append(attempt)
            print(f"{attempt.latency_ms:.0f}ms")

        # 3. Score consistency
        attempts = self._compute_inconsistency(attempts)

        # 4. Adaptive compute
        best = min(attempts, key=lambda a: a.inconsistency_score)
        extra_spawned = 0

        while best.inconsistency_score > self.config.inconsistency_threshold and \
              extra_spawned < self.config.max_extra_attempts:
            alt_agent = agents_to_try[(extra_spawned + 1) % len(agents_to_try)] if len(agents_to_try) > 1 else primary_agent
            print(f"  [extra] {alt_agent} (high temp) …", end=" ", flush=True)
            old_temp = self.sampling_params.temperature
            self.sampling_params.temperature = min(1.0, old_temp + 0.1)
            attempt = self._run_attempt(alt_agent)
            self.sampling_params.temperature = old_temp
            attempts.append(attempt)
            attempts = self._compute_inconsistency(attempts)
            best = min(attempts, key=lambda a: a.inconsistency_score)
            extra_spawned += 1
            print(f"{attempt.latency_ms:.0f}ms")

        # 5. Synthesise
        final_answer = best.extracted_answer or ""
        if not final_answer:
            answers = [a.extracted_answer for a in attempts if a.extracted_answer]
            if answers:
                final_answer = Counter(answers).most_common(1)[0][0]

        raw_confidence = 1.0 / (1.0 + best.inconsistency_score)
        confidence = max(0.0, min(1.0, raw_confidence))

        elapsed = (time.perf_counter() - t0) * 1000

        # Release HF model to free memory
        self._unload_hf_model()

        return LoopResult(
            problem=self.config.problem,
            final_answer=final_answer,
            confidence=confidence,
            attempts=attempts,
            selected_agent=best.agent_name,
            total_time_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="The reasoning problem to solve")
    parser.add_argument("--adapter", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--num-attempts", type=int, default=3)
    parser.add_argument("--max-extra-attempts", type=int, default=3)
    parser.add_argument("--hf-device", default="auto", help="Device for HF hidden state extraction: auto, cpu, cuda:0, etc.")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    # Check for required environment variables
    if "HF_TOKEN" not in os.environ:
        print("warning: HF_TOKEN not set. The model may be gated and fail to download.")
    if "HF_HOME" not in os.environ:
        print("note: HF_HOME not set; model cache defaults to ~/.cache/huggingface.")

    config = LoopConfig(
        problem=args.problem,
        adapter_dir=args.adapter,
        model_id=args.model,
        num_attempts=args.num_attempts,
        max_extra_attempts=args.max_extra_attempts,
        hf_device=args.hf_device,
        output_path=args.output,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    print(f"[loop] problem: {args.problem[:100]}...")
    loop = ConsistencyLoop(config)
    result = loop.solve()

    print(f"\n[result] answer: {result.final_answer}")
    print(f"  confidence: {result.confidence:.3f}")
    print(f"  selected agent: {result.selected_agent}")
    print(f"  attempts: {len(result.attempts)}")
    print(f"  total time: {result.total_time_ms:.0f}ms")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "problem": result.problem,
                "final_answer": result.final_answer,
                "confidence": result.confidence,
                "selected_agent": result.selected_agent,
                "num_attempts": len(result.attempts),
                "total_time_ms": result.total_time_ms,
                "attempts": [
                    {
                        "agent": a.agent_name,
                        "answer": a.extracted_answer,
                        "inconsistency": a.inconsistency_score,
                        "latency_ms": a.latency_ms,
                    }
                    for a in result.attempts
                ],
            }, f, indent=2)
        print(f"  output → {output_path}")


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