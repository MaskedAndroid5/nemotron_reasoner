#!/usr/bin/env python3
"""
agent_router.py — Phase 2: Meta‑Reasoning Agent Router (vLLM‑only)

Classifies a reasoning problem into one or more types from the 10‑agent
taxonomy and returns the best agent(s) to solve it.

Strategy
--------
  • **Few‑shot classification** – Nemotron itself is prompted with the
    list of agents and their role descriptions, then asked to classify
    the problem.  This is the only supported strategy because Nemotron‑3‑Nano
    is a generative model; it cannot be used as an embedding encoder.

Requirements
------------
  • vLLM ≥ 0.12.0 (or 0.11.2 minimum) for full Nemotron‑H LoRA support.
  • GPU with ≥ 40 GB VRAM (BF16) or 24 GB (FP8 variant).
  • Set HF_TOKEN and HF_HOME environment variables for model download.

Usage
-----
  from agent_router import AgentRouter
  router = AgentRouter(model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
  candidates = router.classify("If x + 3 = 7, what is x?")
  # → [("mathematical_reasoning", 1.0), ("logical_deduction", 0.0), ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator
from vllm import LLM, SamplingParams

# Local taxonomy
from reasoning_taxonomy import get_agent, list_agents


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class RouterConfig(BaseModel):
    model_id: str = Field("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=512)
    tensor_parallel_size: int = Field(1, ge=1, description="Number of GPUs for tensor parallelism (1, 2, 4, 8)")
    top_k: int = Field(3, ge=1, le=10, description="Number of candidate agents to return")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

    @validator("tensor_parallel_size")
    def _check_tp(cls, v):
        if v not in (1, 2, 4, 8):
            raise ValueError("tensor_parallel_size must be 1, 2, 4, or 8")
        return v

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Few‑shot classification prompt
# ---------------------------------------------------------------------------
FEWSHOT_TEMPLATE = """\
You are a precise problem classifier.  Given a reasoning problem, determine
which type of reasoning it requires.  Choose EXACTLY ONE agent from the list
below.  Answer with ONLY the agent name, nothing else.

Available agents:
{agent_list}

Problem:
{problem}

Agent name:"""


# ---------------------------------------------------------------------------
# Nemotron‑native router (vLLM only)
# ---------------------------------------------------------------------------
class NemotronRouter:
    """Classify problems by prompting Nemotron via vLLM."""

    def __init__(self, config: RouterConfig):
        self.llm = LLM(
            model=config.model_id,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,       # recommended for reasoning tasks
            top_p=0.95,
            max_tokens=64,
        )
        self.agent_names = list_agents()
        self.agent_descriptions = "\n".join(
            f"  {name}: {get_agent(name).role}"
            for name in self.agent_names
        )
        self._name_to_canonical = {name.lower(): name for name in self.agent_names}

    def classify(self, problem: str) -> List[Tuple[str, float]]:
        """
        Return ranked agents with confidence scores.
        Currently confidence is binary: 1.0 for the predicted agent,
        0.0 for all others.
        """
        prompt = FEWSHOT_TEMPLATE.format(
            agent_list=self.agent_descriptions,
            problem=problem,
        )
        outputs = self.llm.generate([prompt], self.sampling_params)
        result = outputs[0].outputs[0]
        raw_text = result.text.strip()
        predicted = self._resolve_agent(raw_text)

        results = [(predicted, 1.0)]
        for name in self.agent_names:
            if name != predicted:
                results.append((name, 0.0))
        return results

    def _resolve_agent(self, text: str) -> str:
        """Map model output to canonical agent name."""
        text_lower = text.lower().strip()
        # Direct match
        if text_lower in self._name_to_canonical:
            return self._name_to_canonical[text_lower]
        # Substring match
        for name in self.agent_names:
            if name.lower() in text_lower:
                return name
        # Fallback: first agent
        return self.agent_names[0]


# ---------------------------------------------------------------------------
# Unified router API
# ---------------------------------------------------------------------------
class AgentRouter:
    """Top‑level router: loads Nemotron once, exposes classify()."""

    def __init__(self, config: Optional[RouterConfig] = None, **kwargs):
        if config is None:
            config = RouterConfig(**kwargs)
        self.config = config
        self._router = NemotronRouter(config)

    def classify(self, problem: str) -> List[Tuple[str, float]]:
        """Return top‑k agent names with confidence scores."""
        raw = self._router.classify(problem)
        filtered = [
            (name, conf) for name, conf in raw
            if conf >= self.config.confidence_threshold
        ]
        return filtered[:self.config.top_k]


# ---------------------------------------------------------------------------
# CLI — quick test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Agent router – classify a reasoning problem")
    parser.add_argument("--problem", required=True, help="The reasoning problem to classify")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    config = RouterConfig(
        model_id=args.model,
        top_k=args.top_k,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print(f"[router] model={args.model}")
    router = AgentRouter(config)
    results = router.classify(args.problem)

    for name, conf in results:
        agent = get_agent(name)
        print(f"  {name:30s}  confidence={conf:.3f}  {agent.role[:80]}...")


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