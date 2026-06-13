#!/usr/bin/env python3
"""
generate_synthetic_data_v3.py — Phase 1: Async Synthetic Data Pipeline (Nemotron‑native)

Generates high‑quality, sheaf‑structured reasoning traces using **only**
Nemotron‑3‑Nano‑30B via a local vLLM instance.  No external API calls.

Architecture
--------------------------------------------------
  • ProblemSpec / GeneratedExample / FailureReport — Pydantic models that
    enforce the data contract at every boundary.
  • LLMGenerator protocol — the pipeline only knows this interface.
    NemotronGenerator implements it using a local vLLM engine.
  • True concurrency via `asyncio.gather` — all tasks are created upfront
    and executed concurrently, with a `Semaphore` limiting active parallelism
    to `--concurrency` slots.
  • Checkpoint‑with‑integrity — saves progress every CHECKPOINT_INTERVAL
    examples.  On resume reconciles the checkpoint against the actual
    number of valid lines in `examples.jsonl`.  Atomic writes throughout.
  • Structured crash reports — unhandled exceptions produce a JSON file
    with full traceback.

Critical fixes applied (v3.1):
  - Thread‑safe SamplingParams (per‑call copy)
  - Lock around vLLM calls for thread safety
  - Chat‑template prompt formatting for Nemotron
  - Case‑insensitive answer matching
  - Accurate token counting via tokenizer
  - Pydantic V2 compatibility
  - Self‑closing XML tag handling
  - OOM recovery with memory cache clearing

Usage
-----
  python generate_synthetic_data_v3.py \\
      --agent logical_deduction \\
      --num-examples 200 \\
      --output-dir data/logical_deduction \\
      --concurrency 4 \\
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer

# Pydantic V2 compatibility
try:
    from pydantic import model_dump_json as _model_dump_json
except ImportError:
    def _model_dump_json(obj, **kwargs):
        return obj.json(**kwargs)
else:
    def _model_dump_json(obj, **kwargs):
        return obj.model_dump_json(**kwargs)

# Local taxonomy
from reasoning_taxonomy import get_agent, list_agents, generate_prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.8          # high enough for diversity
DEFAULT_TOP_P = 0.95
DEFAULT_CONCURRENCY = 4            # conservative for local vLLM
CHECKPOINT_INTERVAL = 20

BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


# ---------------------------------------------------------------------------
# Pydantic models — the data contract
# ---------------------------------------------------------------------------

class ProblemSpec(BaseModel):
    agent: str
    variables: Dict[str, str] = Field(
        ..., description="Placeholder values for the prompt template"
    )
    ground_truth: str

    @validator("variables")
    def check_required(cls, v, values):
        if "agent" not in values:
            return v
        agent = get_agent(values["agent"])
        required = set(agent.required_variables())
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return v

    class Config:
        extra = "forbid"


class GeneratedExample(BaseModel):
    agent: str
    variables: Dict[str, str]
    ground_truth: str
    generated_trace: str
    extracted_answer: str
    word_count: int
    token_count: int
    xml_valid: bool
    latency_ms: float


class FailureReport(BaseModel):
    index: int
    agent: str
    variables: Optional[Dict[str, str]] = None
    ground_truth: Optional[str] = None
    error_category: str
    detail: str = ""
    last_output_snippet: Optional[str] = None


class RunReport(BaseModel):
    agent: str
    backend: str
    model: str
    requested: int
    generated: int
    failed: int
    failure_categories: Dict[str, int]
    latency_stats: Dict[str, float]
    token_usage: Dict[str, int]
    cost_estimate: float
    elapsed_sec: float
    checkpoint_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Async generator protocol
# ---------------------------------------------------------------------------

class LLMGenerator(Protocol):
    async def generate(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        ...


# ---------------------------------------------------------------------------
# Nemotron‑native back‑end (local vLLM) — thread‑safe version
# ---------------------------------------------------------------------------

class NemotronGenerator:
    """Generate reasoning traces using a local Nemotron instance via vLLM."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
    ):
        from vllm import LLM

        print(f"[nemotron] loading model {model_id} …")
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model_id = model_id
        self._lock = asyncio.Lock()

    async def generate(
        self,
        system: str,
        user: str,
        model: str = "",          # ignored – we always use Nemotron
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> Tuple[str, int, int]:
        """Generate a reasoning trace with thread‑safe vLLM access."""
        from vllm import SamplingParams

        loop = asyncio.get_running_loop()

        # Format prompt using the official chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Create a fresh SamplingParams per call (no race condition)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Thread‑safe generation with a lock
        async with self._lock:
            try:
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate([prompt], sampling_params)
                )
                text = outputs[0].outputs[0].text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError("CUDA OOM during generation – try reducing max_tokens or concurrency")
            except Exception:
                torch.cuda.empty_cache()
                raise

        # Accurate token counts using the tokenizer
        inp_tok = len(self.tokenizer.encode(prompt))
        out_tok = len(self.tokenizer.encode(text))
        return text, inp_tok, out_tok


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_trace(
    text: str,
    ground_truth: str,
    agent_name: str,
    max_tokens: int,
    tokenizer,
) -> Tuple[bool, str, str, int]:
    """Returns (is_valid, extracted_answer, error_category, token_count)."""
    agent = get_agent(agent_name)
    tag_names = list(agent.output_tags.keys())

    # Accurate token count
    tok_count = len(tokenizer.encode(text)) if tokenizer else len(text.split()) * 1.3
    if tok_count > max_tokens * 1.2:
        return False, "", "token_overflow", tok_count

    # XML tag matching (handle self‑closing tags)
    for tag in tag_names:
        # Count opening tags (including self‑closing)
        opens = len(re.findall(rf"<{tag}\b[^>]*/>", text))
        # Count proper opening tags (not self‑closing)
        opens += len(re.findall(rf"<{tag}\b[^>/][^>]*>", text))
        closes = len(re.findall(rf"</{tag}>", text))
        if opens != closes:
            return False, "", "xml_malformed", tok_count

    # Boxed answer
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return False, "", "no_boxed", tok_count
    extracted = matches[-1].strip()

    # Case‑insensitive answer matching
    if extracted.lower() != ground_truth.strip().lower():
        return False, extracted, "answer_mismatch", tok_count

    if "<compatible>" not in text and "<incompatible" not in text:
        return False, extracted, "no_consistency_tag", tok_count

    if agent_name == "code_reasoning" and "<assert" not in text:
        return False, extracted, "missing_assert", tok_count

    return True, extracted, "", tok_count


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

async def generate_one(
    sem: asyncio.Semaphore,
    generator: LLMGenerator,
    agent_name: str,
    problem: ProblemSpec,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    index: int,
    tokenizer,
) -> Tuple[Optional[GeneratedExample], Optional[FailureReport], int, int]:
    """Generate and validate a single example."""
    agent = get_agent(agent_name)
    messages = generate_prompt(agent_name, **problem.variables)
    last_text = ""
    total_inp = 0
    total_out = 0

    async with sem:
        for attempt in range(1, retries + 1):
            t0 = time.perf_counter()
            try:
                text, inp_tok, out_tok = await generator.generate(
                    messages["system"],
                    messages["user"],
                    model="",                # NemotronGenerator ignores this
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                total_inp += inp_tok
                total_out += out_tok
            except RuntimeError as exc:
                if "OOM" in str(exc):
                    fail = FailureReport(
                        index=index,
                        agent=agent_name,
                        variables=problem.variables,
                        ground_truth=problem.ground_truth,
                        error_category="oom",
                        detail="CUDA out of memory during generation",
                    )
                    return None, fail, total_inp, total_out
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                fail = FailureReport(
                    index=index,
                    agent=agent_name,
                    variables=problem.variables,
                    ground_truth=problem.ground_truth,
                    error_category="generator_error",
                    detail=str(exc),
                )
                return None, fail, total_inp, total_out
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate_limit" in exc_str or "429" in exc_str:
                    wait = min(60, 2 ** (attempt + 2))
                    print(f"    [attempt {attempt}] rate limited – waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

                fail = FailureReport(
                    index=index,
                    agent=agent_name,
                    variables=problem.variables,
                    ground_truth=problem.ground_truth,
                    error_category="api_error",
                    detail=str(exc),
                )
                return None, fail, total_inp, total_out

            if text is None:
                await asyncio.sleep(2 ** attempt)
                continue

            last_text = text
            valid, extracted, err_cat, _ = validate_trace(
                text, problem.ground_truth, agent_name, max_tokens, tokenizer
            )
            latency = (time.perf_counter() - t0) * 1000

            if valid:
                ex = GeneratedExample(
                    agent=agent_name,
                    variables=problem.variables,
                    ground_truth=problem.ground_truth,
                    generated_trace=text,
                    extracted_answer=extracted,
                    word_count=len(text.split()),
                    token_count=total_out,
                    xml_valid=True,
                    latency_ms=latency,
                )
                return ex, None, total_inp, total_out
            else:
                if attempt == retries:
                    fail = FailureReport(
                        index=index,
                        agent=agent_name,
                        variables=problem.variables,
                        ground_truth=problem.ground_truth,
                        error_category=err_cat,
                        detail="Max retries exceeded",
                        last_output_snippet=last_text[:200],
                    )
                    return None, fail, total_inp, total_out
                await asyncio.sleep(2 ** attempt)

    fail = FailureReport(
        index=index,
        agent=agent_name,
        error_category="unknown",
        detail="Retry loop exhausted unexpectedly",
    )
    return None, fail, total_inp, total_out


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def _append_jsonl(path: Path, obj: BaseModel) -> bool:
    try:
        with open(path, "a") as f:
            f.write(_model_dump_json(obj) + "\n")
        return True
    except Exception as e:
        print(f"Failed to write to {path}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _verify_checkpoint_integrity(ckpt_path: Path, examples_path: Path) -> int:
    ckpt = _safe_read_json(ckpt_path)
    ckpt_idx = ckpt.get("last_completed_index", 0)

    file_lines = 0
    if examples_path.exists():
        with open(examples_path, "r") as f:
            for line in f:
                if line.strip():
                    file_lines += 1

    if ckpt_idx != file_lines:
        reconciled = max(ckpt_idx, file_lines)
        print(
            f"WARNING: Checkpoint says {ckpt_idx} completed, but "
            f"examples.jsonl has {file_lines} valid lines. "
            f"Resuming from index {reconciled}."
        )
        _safe_write_json(
            ckpt_path,
            {"last_completed_index": reconciled,
             "timestamp": datetime.now().isoformat()},
        )
        return reconciled

    return ckpt_idx


def _count_valid_lines(file_path: Path) -> int:
    cnt = 0
    if file_path.exists():
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    cnt += 1
                except json.JSONDecodeError:
                    pass
    return cnt


def _build_existing_report(
    output_dir: Path, agent_name: str, backend: str, model: str
) -> RunReport:
    n_examples = _count_valid_lines(output_dir / "examples.jsonl")
    n_failures = _count_valid_lines(output_dir / "failures.json")
    return RunReport(
        agent=agent_name,
        backend=backend,
        model=model,
        requested=n_examples + n_failures,
        generated=n_examples,
        failed=n_failures,
        failure_categories={},
        latency_stats={},
        token_usage={},
        cost_estimate=0.0,
        elapsed_sec=0.0,
    )


def _compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < n:
            return sorted_lats[f] + c * (sorted_lats[f + 1] - sorted_lats[f])
        return sorted_lats[f]

    return {
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


async def _warmup_generator(generator: LLMGenerator, tokenizer) -> None:
    print("Warming up generator...", end=" ", flush=True)
    try:
        await generator.generate(
            system="You are a helpful assistant.",
            user="Say 'ready'.",
            model="",
            max_tokens=10,
            temperature=0.0,
            top_p=0.95,
        )
        print("ready.")
    except Exception as e:
        print(f"warmup failed: {e}")


async def run_pipeline(
    agent_name: str,
    problems: List[ProblemSpec],
    generator: LLMGenerator,
    backend: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    retries: int,
    output_dir: Path,
    tokenizer,
) -> RunReport:
    """Execute the generation DAG with true concurrency."""
    sem = asyncio.Semaphore(concurrency)
    examples: List[GeneratedExample] = []
    failures: List[FailureReport] = []
    total_inp = 0
    total_out = 0
    latencies: List[float] = []

    ckpt_path = output_dir / "checkpoint.json"
    examples_path = output_dir / "examples.jsonl"
    failures_path = output_dir / "failures.json"

    start_idx = 0
    if ckpt_path.exists():
        start_idx = _verify_checkpoint_integrity(ckpt_path, examples_path)
        print(f"Resuming from index: {start_idx}")

    pending = problems[start_idx:]
    if not pending:
        print("All examples already completed. Nothing to do.")
        return _build_existing_report(output_dir, agent_name, backend, model)

    tasks = [
        generate_one(
            sem, generator, agent_name, problems[idx],
            max_tokens, temperature, top_p, retries,
            index=idx,
            tokenizer=tokenizer,
        )
        for idx in range(start_idx, len(problems))
    ]

    t_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for offset, result in enumerate(results):
        idx = start_idx + offset
        pct = 100.0 * (idx + 1) / len(problems)

        if isinstance(result, Exception):
            fail = FailureReport(
                index=idx,
                agent=agent_name,
                variables=problems[idx].variables,
                ground_truth=problems[idx].ground_truth,
                error_category="pipeline_error",
                detail=str(result),
            )
            failures.append(fail)
            _append_jsonl(failures_path, fail)
            print(f"  [{idx+1}/{len(problems)} ({pct:.1f}%)] FAIL [pipeline_error]")
            continue

        ex, fail, inp_tok, out_tok = result
        total_inp += inp_tok
        total_out += out_tok

        if ex is not None:
            if _append_jsonl(examples_path, ex):
                examples.append(ex)
                latencies.append(ex.latency_ms)
            print(
                f"  [{idx+1}/{len(problems)} ({pct:.1f}%)] OK "
                f"({ex.word_count} words, {ex.token_count} tokens, {ex.latency_ms:.0f}ms)"
            )
        else:
            failures.append(fail)
            _append_jsonl(failures_path, fail)
            print(f"  [{idx+1}/{len(problems)} ({pct:.1f}%)] FAIL [{fail.error_category}]")

        completed = start_idx + offset + 1
        if completed % CHECKPOINT_INTERVAL == 0:
            _safe_write_json(
                ckpt_path,
                {"last_completed_index": completed,
                 "timestamp": datetime.now().isoformat()},
            )

    elapsed = time.perf_counter() - t_start
    latency_stats = _compute_latency_percentiles(latencies)

    report = RunReport(
        agent=agent_name,
        backend=backend,
        model=model,
        requested=len(problems),
        generated=len(examples),
        failed=len(failures),
        failure_categories=dict(Counter(f.error_category for f in failures)),
        latency_stats=latency_stats,
        token_usage={"input": total_inp, "output": total_out},
        cost_estimate=0.0,   # no API cost for local Nemotron
        elapsed_sec=round(elapsed, 1),
        checkpoint_path=str(ckpt_path),
    )

    completion_path = output_dir / "completion.json"
    _safe_write_json(
        completion_path,
        {
            "completed_at": datetime.now().isoformat(),
            "total_examples": len(problems),
            "generated": len(examples),
        },
    )
    ckpt_path.unlink(missing_ok=True)

    with open(output_dir / "run_report.json", "w") as f:
        f.write(_model_dump_json(report, indent=2))

    return report


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

def _validate_templates(templates: List[Dict[str, str]], agent_name: str) -> None:
    agent = get_agent(agent_name)
    required = agent.required_variables()
    for i, tmpl in enumerate(templates):
        if "ground_truth" not in tmpl:
            raise ValueError(f"Template {i} missing 'ground_truth'")
        vars_keys = {k for k in tmpl if k != "ground_truth"}
        missing = required - vars_keys
        if missing:
            raise ValueError(f"Template {i} missing variables: {missing}")


def _problem_factory(
    agent: str, templates: List[Dict[str, str]], num: int
) -> List[ProblemSpec]:
    specs = []
    for i in range(num):
        tmpl = templates[i % len(templates)]
        variables = {k: v for k, v in tmpl.items() if k != "ground_truth"}
        specs.append(
            ProblemSpec(
                agent=agent,
                variables=variables,
                ground_truth=tmpl["ground_truth"],
            )
        )
    return specs


AGENT_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "logical_deduction": [
        {"premises": "All A are B.\nAll B are C.", "question": "Are all A C?", "ground_truth": "Yes"},
        {"premises": "Some A are B.\nNo B are C.", "question": "Are some A C?", "ground_truth": "No"},
        {"premises": "If it rains, the ground is wet.\nIt is raining.", "question": "Is the ground wet?", "ground_truth": "Yes"},
        {"premises": "All dogs are mammals.\nAll mammals are animals.\nFido is a dog.", "question": "Is Fido an animal?", "ground_truth": "Yes"},
        {"premises": "Either the butler did it or the maid did it.\nThe maid did not do it.", "question": "Did the butler do it?", "ground_truth": "Yes"},
        {"premises": "All birds can fly.\nPenguins are birds.\nPenguins cannot fly.", "question": "Can penguins fly?", "ground_truth": "Inconsistent"},
    ],
    "mathematical_reasoning": [
        {"premises": "x + y = 10\nx - y = 2", "question": "What is x?", "ground_truth": "6"},
        {"premises": "2x + 3 = 7", "question": "Solve for x.", "ground_truth": "2"},
        {"premises": "A rectangle has length 5 and width 3.", "question": "What is its area?", "ground_truth": "15"},
        {"premises": "John has twice as many apples as Mary. Together they have 12.", "question": "How many does Mary have?", "ground_truth": "4"},
        {"premises": "x + y = 5\nx + y = 7", "question": "What is x + y?", "ground_truth": "Inconsistent"},
    ],
    "temporal_spatial": [
        {"premises": "A before B\nB before C", "question": "Is A before C?", "ground_truth": "Yes"},
        {"premises": "X is north of Y\nY is north of Z", "question": "Is X north of Z?", "ground_truth": "Yes"},
        {"premises": "Book on shelf 1. Shelf 1 above shelf 2.", "question": "Is book above shelf 2?", "ground_truth": "Yes"},
        {"premises": "A before B\nB before A", "question": "Is A before B?", "ground_truth": "Inconsistent"},
    ],
    "multi_hop_qa": [
        {"premises": "Fact 1: Paris is the capital of France.\nFact 2: France is in Europe.", "question": "Is Paris in Europe?", "ground_truth": "Yes"},
        {"premises": "Fact 1: Alice is Bob's sister.\nFact 2: Bob is Carol's father.", "question": "Is Alice Carol's aunt?", "ground_truth": "Yes"},
        {"premises": "Fact 1: Tesla makes electric cars.\nFact 2: Model 3 is a Tesla.", "question": "Is Model 3 electric?", "ground_truth": "Yes"},
    ],
    "contradictory_premises": [
        {"premises": "Statement 1: Alice is taller than Bob.\nStatement 2: Bob is taller than Charlie.\nStatement 3: Charlie is taller than Alice.", "question": "Are these statements consistent?", "ground_truth": "Inconsistent"},
        {"premises": "Claim A: All swans are white.\nClaim B: There exists a black swan.", "question": "Are these consistent?", "ground_truth": "Inconsistent"},
    ],
    "incomplete_information": [
        {"premises": "Some birds can fly.\nPenguins are birds.", "question": "Can penguins fly?", "ground_truth": "Cannot be determined"},
        {"premises": "John owns a pet.\nThe pet is either a cat or a dog.", "question": "Is the pet a cat?", "ground_truth": "Cannot be determined"},
    ],
    "iterative_state_transition": [
        {"initial_state": "state[0] = 3", "transition_rule": "state[t+1] = (state[t] * 5 + 1) mod 7", "num_steps": "4", "question": "What is state[4]?", "ground_truth": "0"},
        {"initial_state": "x = 1", "transition_rule": "x <- x + 2", "num_steps": "3", "question": "Final x?", "ground_truth": "7"},
    ],
    "code_reasoning": [
        {"code": "def add(a, b):\n    return a + b\n\nresult = add(2, 3)", "question": "What is the value of result?", "ground_truth": "5"},
        {"code": "x = 5\ny = x * 2\nprint(y)", "question": "What is printed?", "ground_truth": "10"},
    ],
    "causal_reasoning": [
        {"evidence": "1. Ice cream sales increase in summer\n2. Drowning incidents increase in summer\n3. Temperature is high in summer\n4. People swim more when temperature is high", "question": "Does buying ice cream cause drowning?", "ground_truth": "No, confounded by temperature"},
        {"evidence": "1. Smoking increases lung cancer rates.\n2. Lung cancer patients are more likely to have smoked.", "question": "Does smoking cause lung cancer?", "ground_truth": "Yes"},
    ],
    "visual_reasoning": [
        {"description": "A square of side 2 contains a circle touching all four sides.", "question": "What is the area of the circle?", "ground_truth": "π"},
        {"description": "Point A is at (0,0). Point B is at (3,4).", "question": "What is the distance AB?", "ground_truth": "5"},
    ],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", choices=["auto", "force", "fresh"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args()

    if args.list_agents:
        print("Available agents:")
        for name in list_agents():
            a = get_agent(name)
            print(f"  {name}: {a.role[:100]}...")
        return

    try:
        get_agent(args.agent)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    templates = AGENT_TEMPLATES.get(args.agent, [])
    if not templates:
        print(f"No built-in templates for '{args.agent}'. Extend AGENT_TEMPLATES.")
        sys.exit(1)

    _validate_templates(templates, args.agent)
    problems = _problem_factory(args.agent, templates, args.num_examples)

    if args.dry_run:
        print(f"Dry run: would generate {len(problems)} examples using Nemotron (local vLLM).")
        return

    if not torch.cuda.is_available():
        print("CUDA not available – NemotronGenerator requires a GPU.")
        sys.exit(1)

    generator = NemotronGenerator(
        model_id=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = generator.tokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "checkpoint.json"
    completion_path = output_dir / "completion.json"

    if args.resume == "fresh":
        for p in (ckpt_path, completion_path):
            if p.exists():
                p.unlink()
                print(f"Deleted {p.name} (fresh start).")
    elif args.resume == "force":
        if not ckpt_path.exists():
            print("--resume=force but no checkpoint found.")
            sys.exit(1)
        print("Force-resuming from existing checkpoint.")
    elif args.resume == "auto":
        if completion_path.exists():
            print("Completion marker found — all examples already generated. Nothing to do.")
            return
        if ckpt_path.exists():
            print("Checkpoint found — will resume automatically.")

    asyncio.run(_warmup_generator(generator, tokenizer))

    print(
        f"Starting Nemotron‑native pipeline for agent '{args.agent}' "
        f"with {len(problems)} problems, concurrency={args.concurrency}"
    )
    report = asyncio.run(
        run_pipeline(
            agent_name=args.agent,
            problems=problems,
            generator=generator,
            backend="nemotron_vllm",
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=args.concurrency,
            retries=args.retries,
            output_dir=output_dir,
            tokenizer=tokenizer,
        )
    )
    print(
        f"\nPipeline complete. {report.generated} generated, {report.failed} failed."
    )
    print(f"Elapsed: {report.elapsed_sec}s")
    print(f"Report: {output_dir / 'run_report.json'}")


# ---------------------------------------------------------------------------
# Hardened entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL UNHANDLED EXCEPTION: {exc}", file=sys.stderr)
        traceback.print_exc()
        try:
            _parser = argparse.ArgumentParser(add_help=False)
            _parser.add_argument("--output-dir", default="./data")
            _args, _ = _parser.parse_known_args()
            _out = Path(_args.output_dir)
            _out.mkdir(parents=True, exist_ok=True)
            with open(_out / "crash_report.json", "w") as _f:
                json.dump({
                    "error": "Unhandled exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }, _f, indent=2)
            print(f"Crash report written to {_out / 'crash_report.json'}")
        except Exception:
            pass
        sys.exit(2)