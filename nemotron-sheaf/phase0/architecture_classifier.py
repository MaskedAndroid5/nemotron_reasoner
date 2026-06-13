#!/usr/bin/env python3
"""
architecture_classifier.py — self‑configuring module classifier for hybrid LLMs

Classifies every nn.Module in a model into one of:
  attention, moe_expert, mamba, embedding, lm_head, other

using a four‑level fallback from most to least authoritative:
  Level 1 – CONFIG        → model.config (authoritative)
  Level 2 – CLASS_NAME    → Python class name (API contract)
  Level 3 – FINGERPRINT   → forward() signature, parameter shapes
  Level 4 – NAME_HEURISTIC→ module path substrings (last resort)

Each classification carries a confidence level and an evidence string.
Low‑confidence results are surfaced for manual review.

Domain rationale
----------------
- Mamba layers use fused selective‑scan kernels that cannot be wrapped by
  LoRA hooks; they must be excluded from target_modules.
- Fused MoE layers store expert weights as 3D Parameter tensors, not as
  individual nn.Linear submodules.  PEFT ≥ 0.17.0 can target these via
  target_parameters.
- Tied weights (e.g. lm_head / embed_tokens) prevent LoRA attachment.
- vLLM ≥ 0.16.0 is required for Nemotron‑H LoRA inference.

Public API
----------
  classify_module(name, module, model_config) → ModuleClassification
  classify_all_modules(named_modules, model_config) → Dict[str, ModuleClassification]
  summarize_classifications(classifications) → Dict[str, Any]
  is_fused_moe(module) → bool
  get_fused_moe_target_parameters(module, module_path) → List[str]
"""

from __future__ import annotations

import inspect
import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch.nn as nn


# ---------------------------------------------------------------------------
# Pattern registry (single source of truth)
# ---------------------------------------------------------------------------
MAMBA_CLASS_PATTERNS: List[str] = [
    # Core
    "Mamba", "Mamba2", "MambaBlock", "MambaLayer", "MambaCore",
    # Variants
    "MambaTransformer", "MambaVision",
    "SelectiveScan", "SelectiveStateSpace", "SSM", "StateSpaceModel",
    # Library‑specific
    "LlamaMamba", "LlamaMambaBlock", "GPT2Mamba", "VisionMamba", "MambaUnet",
]

ATTENTION_CLASS_PATTERNS: List[str] = [
    "SelfAttention", "MultiHeadAttention", "MultiheadAttention",
    "GroupedQueryAttention", "GQA", "FlashAttention",
]

# Broader Attention check (excludes adapters/pools)
def _is_core_attention_class(class_name: str) -> bool:
    if any(pat in class_name for pat in ATTENTION_CLASS_PATTERNS):
        return True
    if "Attention" in class_name:
        exclude = {"Pool", "Dropout", "Adapter", "Wrapper", "Gate", "Mask"}
        return not any(ex in class_name for ex in exclude)
    return False


MOE_CLASS_PATTERNS: List[str] = [
    "MoE", "MixtureOfExperts", "MoELayer", "SparseMoE",
    "ExpertLayer", "SwitchTransformers", "MoERouter",
    "FusedMoE", "MoEModule",
]

ROUTER_CLASS_PATTERNS: List[str] = [
    "Router", "Gate", "Dispatcher", "MoERouter",
    "SwitchRouter", "TopKRouter", "ExpertRouter",
]

# Signature parameters (need ≥2 matches to classify)
MAMBA_SIGNATURE_PARAMS: set = {"dt", "delta", "selective_scan", "ssm_state"}
ATTENTION_SIGNATURE_PARAMS: set = {"attention_mask", "attn_mask", "key_padding_mask"}

# Name‑based fallbacks (Level 4)
MAMBA_NAME_PATTERNS: List[str] = ["mamba", "ssm", "selective_scan", "conv1d"]
ATTENTION_NAME_PATTERNS: List[str] = ["attention", "attn", "self_attn"]
MOE_NAME_PATTERNS: List[str] = ["moe", "expert", "router"]

DEFAULT_MOE_EXPERT_MIN_COUNT: int = 4

# Config keys for Level 1
_LAYER_TYPE_LIST_KEYS: List[str] = [
    "layer_types", "layer_type", "block_types", "layers_block_type",
]
_LAYER_TYPE_MAP_KEYS: Dict[str, List[str]] = {
    "mamba":      ["mamba_layers", "ssm_layers", "selective_scan_layers"],
    "attention":  ["attention_layers", "attn_layers", "self_attention_layers"],
    "moe_expert": ["moe_layers", "expert_layers", "mixture_layers"],
}
_HYBRID_PATTERN_MAP: Dict[str, str] = {
    "M": "mamba", "S": "mamba",
    "*": "attention", "A": "attention", "H": "attention",
    "-": "mlp", "F": "mlp",
    "E": "moe_expert", "G": "moe_expert",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class ClassificationConfidence(Enum):
    CONFIG = 1          # model.config
    CLASS_NAME = 2      # Python class
    FINGERPRINT = 3     # behavioural inspection
    NAME_HEURISTIC = 4  # module path substring


# Default: only NAME_HEURISTIC is low confidence
_LOW_CONFIDENCE_THRESHOLD: set = {ClassificationConfidence.NAME_HEURISTIC}


class ModuleClassification:
    __slots__ = ("category", "confidence", "evidence")

    def __init__(self, category: str, confidence: ClassificationConfidence, evidence: str):
        self.category = category
        self.confidence = confidence
        self.evidence = evidence

    def __repr__(self) -> str:
        return f"{self.category} ({self.confidence.name}: {self.evidence})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_module(
    name: str,
    module: nn.Module,
    model_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
) -> ModuleClassification:
    """Classify a single module.  model_config can be a dict or HuggingFace config."""
    config_dict = _normalise_config(model_config) if model_config is not None else None

    if config_dict is not None:
        result = _classify_from_config(name, module, config_dict)
        if result is not None:
            return result

    result = _classify_from_class(name, module)
    if result is not None:
        return result

    result = _classify_from_fingerprint(name, module, config_dict)
    if result is not None:
        return result

    return _classify_from_name(name, module)


def classify_all_modules(
    named_modules: Iterable[Tuple[str, nn.Module]],
    model_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
) -> Dict[str, ModuleClassification]:
    """Classify every module in an iterable of (name, module)."""
    return {
        name: classify_module(name, module, model_config)
        for name, module in named_modules
    }


def summarize_classifications(
    classifications: Dict[str, ModuleClassification],
    low_confidence_threshold: Optional[set] = None,
) -> Dict[str, Any]:
    """Produce a summary with per‑category counts and low‑confidence items."""
    if low_confidence_threshold is None:
        low_confidence_threshold = _LOW_CONFIDENCE_THRESHOLD

    by_category: Dict[str, int] = defaultdict(int)
    by_confidence: Dict[str, int] = defaultdict(int)
    low_confidence: List[Dict[str, str]] = []

    for path, cls in classifications.items():
        by_category[cls.category] += 1
        by_confidence[cls.confidence.name] += 1
        if cls.confidence in low_confidence_threshold:
            low_confidence.append({
                "path": path,
                "category": cls.category,
                "confidence": cls.confidence.name,
                "evidence": cls.evidence,
            })

    return {
        "total_modules": len(classifications),
        "by_category": dict(by_category),
        "by_confidence": dict(by_confidence),
        "low_confidence_checks": low_confidence,
    }


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------
def _normalise_config(config: Union[Dict[str, Any], "PretrainedConfig"]) -> Dict[str, Any]:
    """Accept a HuggingFace PretrainedConfig or plain dict, return a dict."""
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {
        k: getattr(config, k)
        for k in dir(config)
        if not k.startswith("_") and not callable(getattr(config, k))
    }


# ---------------------------------------------------------------------------
# Level 1 – Config
# ---------------------------------------------------------------------------
def _classify_from_config(
    name: str, module: nn.Module, config: Dict[str, Any]
) -> Optional[ModuleClassification]:
    layer_idx = _extract_layer_index(name)
    if layer_idx is None:
        return None

    # Explicit per‑index lists (with validation)
    for key in _LAYER_TYPE_LIST_KEYS:
        if key not in config:
            continue
        types = config[key]
        if not isinstance(types, (list, tuple)):
            continue
        if layer_idx >= len(types):
            continue
        layer_type = types[layer_idx]
        if not isinstance(layer_type, str):
            continue
        return ModuleClassification(
            _normalize_category(layer_type),
            ClassificationConfidence.CONFIG,
            f"config.{key}[{layer_idx}] = '{layer_type}'",
        )

    # Per‑type sets / lists
    for category, cfg_keys in _LAYER_TYPE_MAP_KEYS.items():
        for cfg_key in cfg_keys:
            layer_collection = config.get(cfg_key)
            if layer_collection is None:
                continue
            try:
                if layer_idx in layer_collection:
                    return ModuleClassification(
                        category,
                        ClassificationConfidence.CONFIG,
                        f"config.{cfg_key} contains layer {layer_idx}",
                    )
            except TypeError:
                pass

    # hybrid_override_pattern (Nemotron‑H)
    if "hybrid_override_pattern" in config and "num_hidden_layers" in config:
        pattern = config["hybrid_override_pattern"]
        num_layers = config["num_hidden_layers"]
        if isinstance(pattern, str) and layer_idx < min(num_layers, len(pattern)):
            char = pattern[layer_idx]
            if char in _HYBRID_PATTERN_MAP:
                layer_type = _HYBRID_PATTERN_MAP[char]
                return ModuleClassification(
                    _normalize_category(layer_type),
                    ClassificationConfidence.CONFIG,
                    f"config.hybrid_override_pattern[{layer_idx}] = '{char}' → {layer_type}",
                )
            else:
                # Unknown character — flag for review
                return ModuleClassification(
                    "other",
                    ClassificationConfidence.FINGERPRINT,
                    f"Unknown hybrid pattern character: '{char}' at layer {layer_idx}",
                )

    return None


# ---------------------------------------------------------------------------
# Level 2 – Class name
# ---------------------------------------------------------------------------
def _classify_from_class(name: str, module: nn.Module) -> Optional[ModuleClassification]:
    class_name = type(module).__name__

    if isinstance(module, nn.Embedding):
        if not any(p in class_name for p in ("Rotary", "Positional", "Patch", "Sinusoidal")):
            return ModuleClassification(
                "embedding", ClassificationConfidence.CLASS_NAME,
                f"Class '{class_name}' is nn.Embedding",
            )

    for pat in MAMBA_CLASS_PATTERNS:
        if pat in class_name:
            return ModuleClassification("mamba", ClassificationConfidence.CLASS_NAME,
                                        f"Class '{class_name}' matches Mamba pattern '{pat}'")

    if _is_core_attention_class(class_name):
        return ModuleClassification("attention", ClassificationConfidence.CLASS_NAME,
                                    f"Class '{class_name}' is a core attention layer")

    for pat in MOE_CLASS_PATTERNS:
        if pat in class_name:
            return ModuleClassification("moe_expert", ClassificationConfidence.CLASS_NAME,
                                        f"Class '{class_name}' matches MoE pattern '{pat}'")

    for pat in ROUTER_CLASS_PATTERNS:
        if pat in class_name:
            return ModuleClassification("moe_router", ClassificationConfidence.CLASS_NAME,
                                        f"Class '{class_name}' matches MoE router pattern '{pat}'")

    return None


# ---------------------------------------------------------------------------
# Level 3 – Fingerprinting
# ---------------------------------------------------------------------------
def _classify_from_fingerprint(
    name: str,
    module: nn.Module,
    model_config: Optional[Dict[str, Any]] = None,
) -> Optional[ModuleClassification]:
    # 1. Mamba attributes
    if hasattr(module, "dt_bias") or hasattr(module, "delta_rank"):
        return ModuleClassification(
            "mamba", ClassificationConfidence.FINGERPRINT,
            "Module has dt_bias or delta_rank (Mamba selective scan)",
        )

    # Collect children once
    children = list(module.children())

    # 2. Fused MoE: 3D param WITHOUT nn.Linear children, minimum 4 experts
    has_linear = any(isinstance(c, nn.Linear) for c in children)
    if not has_linear:
        for pname, param in module.named_parameters(recurse=False):
            if param.dim() == 3 and param.shape[0] >= 4:   # at least 4 experts
                return ModuleClassification(
                    "moe_expert", ClassificationConfidence.FINGERPRINT,
                    f"Parameter '{pname}' shape {tuple(param.shape)} "
                    "(3D fused expert tensor, no accessible nn.Linear children)",
                )

    # 3. Forward signature inspection (require ≥2 matching parameters)
    if hasattr(module, "forward"):
        try:
            sig = inspect.signature(module.forward)
            param_names = {p for p in sig.parameters if p != "self"}
            mamba_match = MAMBA_SIGNATURE_PARAMS & param_names
            if len(mamba_match) >= 2:
                return ModuleClassification(
                    "mamba", ClassificationConfidence.FINGERPRINT,
                    f"Forward signature contains Mamba parameters: {mamba_match}",
                )
            attn_match = ATTENTION_SIGNATURE_PARAMS & param_names
            if len(attn_match) >= 2:
                return ModuleClassification(
                    "attention", ClassificationConfidence.FINGERPRINT,
                    f"Forward signature contains attention parameters: {attn_match}",
                )
        except (ValueError, TypeError, AttributeError):
            pass

    # 4. MoE structural pattern (parallel Linear children)
    if not children:
        return None
    moe_min = _get_moe_min_count(model_config)
    if _has_parallel_linear_children(module, children=children, min_count=moe_min):
        return ModuleClassification(
            "moe_expert", ClassificationConfidence.FINGERPRINT,
            f"Module contains >= {moe_min} parallel Linear children (MoE pattern)",
        )

    return None


def _get_moe_min_count(model_config: Optional[Dict[str, Any]]) -> int:
    if model_config is not None:
        return model_config.get("moe_expert_min_count", DEFAULT_MOE_EXPERT_MIN_COUNT)
    return DEFAULT_MOE_EXPERT_MIN_COUNT


def _has_parallel_linear_children(
    module: nn.Module,
    min_count: Optional[int] = None,
    children: Optional[List[nn.Module]] = None,
) -> bool:
    if min_count is None:
        min_count = DEFAULT_MOE_EXPERT_MIN_COUNT
    if children is None:
        children = list(module.children())

    shapes: List[Tuple[int, ...]] = []
    for child in children:
        if isinstance(child, nn.Linear):
            w = getattr(child, "weight", None)
            if w is not None:
                shapes.append(tuple(w.shape))

    if len(shapes) < min_count:
        return False
    shape_counts = Counter(shapes)
    return shape_counts.most_common(1)[0][1] >= min_count


# ---------------------------------------------------------------------------
# Level 4 – Name heuristics
# ---------------------------------------------------------------------------
def _classify_from_name(name: str, module: nn.Module) -> ModuleClassification:
    n = name.lower()
    for pat in MAMBA_NAME_PATTERNS:
        if pat in n:
            return ModuleClassification("mamba", ClassificationConfidence.NAME_HEURISTIC,
                                        f"Path contains Mamba indicator '{pat}'")
    for pat in ATTENTION_NAME_PATTERNS:
        if pat in n:
            return ModuleClassification("attention", ClassificationConfidence.NAME_HEURISTIC,
                                        f"Path contains attention indicator '{pat}'")
    for pat in MOE_NAME_PATTERNS:
        if pat in n:
            return ModuleClassification("moe_expert", ClassificationConfidence.NAME_HEURISTIC,
                                        f"Path contains MoE indicator '{pat}'")
    if "embed" in n:
        return ModuleClassification("embedding", ClassificationConfidence.NAME_HEURISTIC,
                                    f"Path contains 'embed'")
    if "lm_head" in n:
        return ModuleClassification("lm_head", ClassificationConfidence.NAME_HEURISTIC,
                                    f"Path contains 'lm_head'")
    return ModuleClassification("other", ClassificationConfidence.NAME_HEURISTIC,
                                "No pattern matched")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_layer_index(name: str) -> Optional[int]:
    matches = re.findall(r"(?:layers?|blocks?|stages?)\.(\d+)", name)
    return int(matches[-1]) if matches else None


def _normalize_category(category: str) -> str:
    cat = category.lower().strip()
    mapping = {
        "mamba": "mamba", "mamba2": "mamba", "ssm": "mamba",
        "selective_scan": "mamba", "state_space": "mamba",
        "attention": "attention", "attn": "attention",
        "self_attn": "attention", "transformer": "attention",
        "moe": "moe_expert", "expert": "moe_expert",
        "mixture": "moe_expert", "sparse": "moe_expert",
        "embedding": "embedding", "embed": "embedding",
        "lm_head": "lm_head", "head": "lm_head",
        "mlp": "other",
    }
    return mapping.get(cat, cat)


# ---------------------------------------------------------------------------
# Fused MoE helpers
# ---------------------------------------------------------------------------
def is_fused_moe(module: nn.Module) -> bool:
    """Return True if module is a fused MoE container (multiple 3D experts, no nn.Linear children)."""
    for child in module.children():
        if isinstance(child, nn.Linear):
            return False
    param_3d_count = sum(1 for p in module.parameters(recurse=False) if p.dim() == 3)
    if param_3d_count == 0:
        return False
    for param in module.parameters(recurse=False):
        if param.dim() == 3 and param.shape[0] >= 4:
            return True
    return False


def get_fused_moe_target_parameters(module: nn.Module, module_path: str) -> List[str]:
    """Return full dotted parameter paths suitable for PEFT's target_parameters."""
    paths = []
    for pname, param in module.named_parameters(recurse=False):
        if param.dim() == 3:
            paths.append(f"{module_path}.{pname}")
    return paths


# ---------------------------------------------------------------------------
# Tied weight detection
# ---------------------------------------------------------------------------
def detect_tied_weights(model: nn.Module) -> Dict[str, List[str]]:
    """Return groups of module paths that share the same parameter tensor."""
    param_to_modules: Dict[int, List[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        param_to_modules[id(param)].append(name)
    return {f"group_{i}": paths for i, (pid, paths) in enumerate(param_to_modules.items()) if len(paths) > 1}


# ---------------------------------------------------------------------------
# Self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Pattern registry loaded. Use 01_module_coverage.py to run the classifier.")