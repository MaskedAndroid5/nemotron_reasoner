#!/usr/bin/env python3
"""
lora_config_loader.py — Phase 2: LoRA Configuration Loader
===========================================================
Loads, validates, and materialises the LoRA configuration produced by
Phase 0 (01_module_coverage.py → lora_config_safe.yaml).

Responsibilities:
  • Parse lora_config_safe.yaml with strict schema validation (Pydantic).
  • Verify that every target_module and target_parameter exists in the
    live model before training begins — fail loudly if not.
  • Construct a PEFT LoraConfig ready for get_peft_model().
  • Expose a typed LoRASettings dataclass consumed by train_lora.py.

Public API:
  • load_lora_settings(yaml_path, model) -> LoRASettings
  • build_peft_config(settings)          -> peft.LoraConfig
  • verify_target_modules(settings, model) -> VerificationReport
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Pydantic schema for lora_config_safe.yaml
# ---------------------------------------------------------------------------

class LoRAYAMLSchema(BaseModel):
    """
    Mirrors the structure written by 01_module_coverage.py.
    All fields are validated at load time — missing or malformed
    configs raise immediately with a precise error message.
    """

    r: int = Field(..., ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(..., ge=1, description="LoRA alpha (scaling)")
    lora_dropout: float = Field(0.0, ge=0.0, le=0.5)
    bias: str = Field("none")
    task_type: str = Field("CAUSAL_LM")
    target_modules: List[str] = Field(default_factory=list)
    target_parameters: List[str] = Field(default_factory=list)

    model_id: Optional[str] = None
    verified_at: Optional[str] = None
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None

    @validator("bias")
    def bias_valid(cls, v: str) -> str:
        allowed = {"none", "all", "lora_only"}
        if v not in allowed:
            raise ValueError(f"bias must be one of {allowed}, got '{v}'")
        return v

    @validator("task_type")
    def task_type_valid(cls, v: str) -> str:
        allowed = {"CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS", "SEQ_CLS"}
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}, got '{v}'")
        return v

    @validator("target_modules", "target_parameters", each_item=True)
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target_modules/target_parameters entries must be non-empty")
        return v.strip()

    @validator("target_parameters", always=True)
    def at_least_one_target(cls, v: List[str], values: Dict[str, Any]) -> List[str]:
        modules = values.get("target_modules", [])
        if not modules and not v:
            raise ValueError(
                "At least one of target_modules or target_parameters must be non-empty"
            )
        return v

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Typed settings object consumed by the rest of phase2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoRASettings:
    """
    Immutable, fully-validated LoRA configuration.
    Passed to build_peft_config() and train_lora.py.
    """
    rank: int
    alpha: int
    dropout: float
    bias: str
    task_type: str
    target_modules: List[str]
    target_parameters: List[str]

    yaml_path: Path
    model_id: Optional[str] = None

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.alpha / self.rank

    @property
    def has_target_modules(self) -> bool:
        return len(self.target_modules) > 0

    @property
    def has_target_parameters(self) -> bool:
        return len(self.target_parameters) > 0

    def summary(self) -> str:
        lines = [
            f"LoRA Configuration  ({self.yaml_path.name})",
            f"  rank={self.rank}  alpha={self.alpha}  "
            f"scale={self.scaling:.3f}  dropout={self.dropout}",
            f"  bias={self.bias}  task_type={self.task_type}",
        ]
        if self.target_modules:
            lines.append(f"  target_modules ({len(self.target_modules)}): "
                         f"{self.target_modules}")
        if self.target_parameters:
            lines.append(f"  target_parameters ({len(self.target_parameters)}): "
                         f"{self.target_parameters}")
        if self.model_id:
            lines.append(f"  model_id: {self.model_id}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------

@dataclass
class VerificationReport:
    """Result of verify_target_modules()."""
    passed: bool
    verified_modules: List[str] = field(default_factory=list)
    verified_parameters: List[str] = field(default_factory=list)
    missing_modules: List[str] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            msg = "\n".join(self.errors)
            raise RuntimeError(
                f"LoRA target verification failed:\n{msg}"
            )

    def summary(self) -> str:
        lines = ["LoRA Target Verification"]
        lines.append(f"  passed: {self.passed}")
        if self.verified_modules:
            lines.append(f"  verified_modules ({len(self.verified_modules)}): "
                         f"{self.verified_modules}")
        if self.verified_parameters:
            lines.append(f"  verified_parameters ({len(self.verified_parameters)}): "
                         f"{self.verified_parameters}")
        if self.missing_modules:
            lines.append(f"  MISSING modules: {self.missing_modules}")
        if self.missing_parameters:
            lines.append(f"  MISSING parameters: {self.missing_parameters}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_lora_settings(
    yaml_path: Path | str,
    model: Optional[Any] = None,
    *,
    verify: bool = True,
) -> LoRASettings:
    """
    Load and validate lora_config_safe.yaml produced by Phase 0.

    Args:
        yaml_path: Path to lora_config_safe.yaml.
        model:     Optional live model. If provided and verify=True,
                   every target_module/parameter is checked against
                   the model's named modules/parameters.
        verify:    Whether to run verify_target_modules(). Default True.
                   Requires model to be provided.

    Returns:
        LoRASettings — immutable, typed configuration object.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError:        If YAML structure is invalid or verify=True
                           but no model provided.
        RuntimeError:      If target module verification fails.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"lora_config_safe.yaml not found at {yaml_path}. "
            f"Run phase0/01_module_coverage.py first."
        )

    raw = yaml.safe_load(yaml_path.read_text())
    if raw is None:
        raise ValueError(f"{yaml_path} is empty.")

    # Support both flat and nested {"lora": {...}} formats
    lora_block = raw.get("lora", raw)

    try:
        schema = LoRAYAMLSchema(**lora_block)
    except Exception as exc:
        raise ValueError(
            f"Invalid lora_config_safe.yaml at {yaml_path}:\n{exc}"
        ) from exc

    settings = LoRASettings(
        rank=schema.r,
        alpha=schema.lora_alpha,
        dropout=schema.lora_dropout,
        bias=schema.bias,
        task_type=schema.task_type,
        target_modules=list(schema.target_modules),
        target_parameters=list(schema.target_parameters),
        yaml_path=yaml_path,
        model_id=schema.model_id,
    )

    if verify:
        if model is None:
            raise ValueError(
                "verify=True requires a model instance. "
                "Pass a model or set verify=False."
            )
        report = verify_target_modules(settings, model)
        print(report.summary())
        report.raise_if_failed()

    return settings


def verify_target_modules(
    settings: LoRASettings,
    model: Any,
) -> VerificationReport:
    """
    Verify that every target_module and target_parameter in settings
    exists in the live model.

    target_modules are matched against model.named_modules() — the
    check is suffix-based (e.g. "q_proj" matches any layer ending in
    ".q_proj"), consistent with how PEFT resolves them.

    target_parameters are matched against model.named_parameters() —
    exact substring match, also consistent with PEFT.

    Args:
        settings: Validated LoRASettings.
        model:    The loaded base model (nn.Module).

    Returns:
        VerificationReport with per-target pass/fail status.
    """
    report = VerificationReport(passed=True)

    all_module_names = {name for name, _ in model.named_modules()}
    all_param_names  = {name for name, _ in model.named_parameters()}

    for module_key in settings.target_modules:
        matched = any(
            name == module_key or name.endswith(f".{module_key}")
            for name in all_module_names
        )
        if matched:
            report.verified_modules.append(module_key)
        else:
            report.missing_modules.append(module_key)
            report.errors.append(
                f"target_module '{module_key}' not found in model. "
                f"Check 01_module_coverage.py output."
            )

    for param_path in settings.target_parameters:
        matched = any(param_path in name for name in all_param_names)
        if matched:
            report.verified_parameters.append(param_path)
        else:
            report.missing_parameters.append(param_path)
            report.errors.append(
                f"target_parameter '{param_path}' not found in model parameters. "
                f"Check 01_module_coverage.py output."
            )

    report.passed = (
        len(report.missing_modules) == 0
        and len(report.missing_parameters) == 0
    )
    return report


def build_peft_config(settings: LoRASettings) -> "peft.LoraConfig":
    """
    Construct a PEFT LoraConfig from validated LoRASettings.

    Args:
        settings: Output of load_lora_settings().

    Returns:
        peft.LoraConfig ready for get_peft_model(model, config).

    Raises:
        ImportError: If PEFT is not installed.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for build_peft_config(). "
            "Install with: pip install peft>=0.17.0"
        ) from exc

    task_type_map = {
        "CAUSAL_LM":    TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS":    TaskType.TOKEN_CLS,
        "SEQ_CLS":      TaskType.SEQ_CLS,
    }
    task_type = task_type_map.get(settings.task_type, TaskType.CAUSAL_LM)

    kwargs: Dict[str, Any] = dict(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        task_type=task_type,
    )

    if settings.has_target_modules:
        kwargs["target_modules"] = settings.target_modules

    if settings.has_target_parameters:
        kwargs["target_parameters"] = settings.target_parameters

    return LoraConfig(**kwargs)


# ---------------------------------------------------------------------------
# CLI — diagnostic mode
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Load and validate lora_config_safe.yaml"
    )
    parser.add_argument(
        "--config", default="phase0_results/lora_config_safe.yaml",
        help="Path to lora_config_safe.yaml"
    )
    parser.add_argument(
        "--model", default=None,
        help="Optional model ID to verify targets against live model"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip live model verification"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_path = Path(args.config)

    print(f"Loading: {yaml_path}")

    model = None
    if args.model and not args.no_verify:
        print(f"Loading model for verification: {args.model}")
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    try:
        settings = load_lora_settings(
            yaml_path,
            model=model,
            verify=(model is not None and not args.no_verify),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(settings.summary())

    if model is not None:
        report = verify_target_modules(settings, model)
        print(report.summary())
        if not report.passed:
            sys.exit(1)

    print("\nBuilding PEFT config...")
    try:
        peft_cfg = build_peft_config(settings)
        print(f"  LoraConfig: r={peft_cfg.r} alpha={peft_cfg.lora_alpha} "
              f"modules={peft_cfg.target_modules}")
        print("ok")
    except ImportError as exc:
        print(f"warning: {exc} (PEFT not installed, skipping config build)")

    sys.exit(0)


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
        try:
            import json
            _out = Path("phase0_results")
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