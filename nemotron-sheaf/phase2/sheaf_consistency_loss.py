#!/usr/bin/env python3
"""
sheaf_consistency_loss.py — Phase 2: Sheaf‑Theoretic Consistency Loss
======================================================================
Implements a differentiable consistency penalty grounded in cellular
sheaf theory over the hypergraph of reasoning claims.

Safety guarantees (v3.2):
  • Spectral regularizer guard — skips Fiedler computation when the
    Laplacian norm is near‑zero (perfect consistency), preventing
    noisy gradients.
  • Out‑of‑order claim detection — warns when an overlap has no
    preceding claims, indicating malformed XML traces.
  • All operations are differentiable and GPU‑resident.

Public API
----------
  SheafLossConfig          — Pydantic config with hyperparameters
  SheafProjections         — nn.Module: restriction maps (per v‑e pair)
  SheafConsistencyLoss     — nn.Module: the full loss
  TagPositionExtractor     — utility: locate tag token positions
  build_sheaf_loss(config) — factory function
"""

from __future__ import annotations

import inspect
import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Pattern Registry (same as architecture_classifier.py for standalone use)
# ---------------------------------------------------------------------------
MAMBA_CLASS_PATTERNS: List[str] = [
    "Mamba", "Mamba2", "MambaBlock", "MambaLayer", "MambaCore",
    "MambaTransformer", "MambaVision",
    "SelectiveScan", "SelectiveStateSpace", "SSM", "StateSpaceModel",
    "LlamaMamba", "LlamaMambaBlock", "GPT2Mamba", "VisionMamba", "MambaUnet",
]

ATTENTION_CLASS_PATTERNS: List[str] = [
    "SelfAttention", "MultiHeadAttention", "MultiheadAttention",
    "GroupedQueryAttention", "GQA", "FlashAttention",
]

MOE_CLASS_PATTERNS: List[str] = [
    "MoE", "MixtureOfExperts", "MoELayer", "SparseMoE",
    "ExpertLayer", "SwitchTransformers", "MoERouter",
    "FusedMoE", "MoEModule",
]

ROUTER_CLASS_PATTERNS: List[str] = [
    "Router", "Gate", "Dispatcher", "MoERouter",
    "SwitchRouter", "TopKRouter", "ExpertRouter",
]

MAMBA_SIGNATURE_PARAMS: set = {"dt", "delta", "selective_scan", "ssm_state"}
ATTENTION_SIGNATURE_PARAMS: set = {"attention_mask", "attn_mask", "key_padding_mask"}

MAMBA_NAME_PATTERNS: List[str] = ["mamba", "ssm", "selective_scan", "conv1d"]
ATTENTION_NAME_PATTERNS: List[str] = ["attention", "attn", "self_attn"]
MOE_NAME_PATTERNS: List[str] = ["moe", "expert", "router"]

DEFAULT_MOE_EXPERT_MIN_COUNT: int = 4

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
# Classification data structures (for standalone use)
# ---------------------------------------------------------------------------
class ClassificationConfidence(Enum):
    CONFIG = 1
    CLASS_NAME = 2
    FINGERPRINT = 3
    NAME_HEURISTIC = 4


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
# Sheaf configuration
# ---------------------------------------------------------------------------
class SheafLossConfig(BaseModel):
    """Validated hyperparameters for the sheaf consistency loss."""

    hidden_dim: int = Field(4096, ge=64)
    edge_dim: int = Field(256, ge=16)
    lambda_compatible: float = Field(0.1, ge=0.0)
    lambda_incompatible: float = Field(0.05, ge=0.0)
    lambda_spectral: float = Field(0.01, ge=0.0)
    incompatible_margin: float = Field(1.0, ge=0.0)
    num_layers_to_average: int = Field(4, ge=1, le=64)
    fiedler_power_iters: int = Field(10, ge=3, le=50)
    fiedler_target: float = Field(0.1, ge=0.0)
    eps: float = Field(1e-8, gt=0.0)

    @validator("edge_dim")
    def edge_dim_le_hidden(cls, v, values):
        if "hidden_dim" in values and v > values["hidden_dim"]:
            raise ValueError(f"edge_dim ({v}) must be ≤ hidden_dim")
        return v

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Tag position extractor
# ---------------------------------------------------------------------------
class TagPositionExtractor:
    COMPATIBLE_TAGS   = ["<compatible>"]
    INCOMPATIBLE_TAGS = ["<incompatible"]
    CLAIM_TAGS        = ["<claim>", "<claim "]
    OVERLAP_TAGS      = ["<overlap>", "<overlap "]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: Dict[str, List[int]] = {}

    def _encode_tag(self, tag: str) -> List[int]:
        if tag not in self._cache:
            self._cache[tag] = self.tokenizer.encode(tag, add_special_tokens=False)
        return self._cache[tag]

    def find_positions(self, input_ids: torch.Tensor, tag_strings: List[str]) -> List[int]:
        positions = []
        ids = input_ids.tolist()
        n = len(ids)
        for tag in tag_strings:
            tag_ids = self._encode_tag(tag)
            if not tag_ids:
                continue
            first_tok, tlen = tag_ids[0], len(tag_ids)
            for i in range(n - tlen + 1):
                if ids[i] == first_tok and ids[i:i + tlen] == tag_ids:
                    positions.append(i)
        return sorted(set(positions))

    def extract_all(self, input_ids: torch.Tensor) -> Tuple[List[int], List[int], List[int], List[int]]:
        return (
            self.find_positions(input_ids, self.CLAIM_TAGS),
            self.find_positions(input_ids, self.COMPATIBLE_TAGS),
            self.find_positions(input_ids, self.INCOMPATIBLE_TAGS),
            self.find_positions(input_ids, self.OVERLAP_TAGS),
        )


# ---------------------------------------------------------------------------
# Restriction maps
# ---------------------------------------------------------------------------
class SheafProjections(nn.Module):
    def __init__(self, config: SheafLossConfig, max_vertex_degree: int = 16):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.edge_dim   = config.edge_dim
        self.vertex_proj = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.edge_dim, bias=False)
            for _ in range(max_vertex_degree)
        ])
        self.edge_proj   = nn.Linear(config.hidden_dim, config.edge_dim, bias=False)
        self.vertex_norm = nn.LayerNorm(config.hidden_dim)
        self.edge_norm   = nn.LayerNorm(config.hidden_dim)
        self._init_isometric()

    def _init_isometric(self):
        for linear in self.vertex_proj:
            W = torch.randn(self.hidden_dim, self.edge_dim)
            U, _, _ = torch.linalg.svd(W, full_matrices=False)
            linear.weight.data.copy_(U.T)
        W = torch.randn(self.hidden_dim, self.edge_dim)
        U, _, _ = torch.linalg.svd(W, full_matrices=False)
        self.edge_proj.weight.data.copy_(U.T)

    def project_vertex(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        return self.vertex_proj[idx](self.vertex_norm(h))

    def project_edge(self, h: torch.Tensor) -> torch.Tensor:
        return self.edge_proj(self.edge_norm(h))


# ---------------------------------------------------------------------------
# Spectral regulariser
# ---------------------------------------------------------------------------
def _build_incidence_matrix(
    n_vertices: int,
    hyperedges: List[List[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_edges = len(hyperedges)
    B = torch.zeros(n_edges, n_vertices, device=device, dtype=dtype)
    for e_idx, verts in enumerate(hyperedges):
        for pos, v in enumerate(verts):
            if v < n_vertices:
                B[e_idx, v] = 1.0 if pos == 0 else -1.0
    return B


def _approximate_fiedler(L: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
    n = L.shape[0]
    if n < 2:
        return torch.zeros(1, device=L.device, dtype=L.dtype)
    v = torch.randn(n, 1, device=L.device, dtype=L.dtype)
    v = v / (v.norm() + eps)
    for _ in range(num_iters):
        v = L @ v
        lam = v.norm()
        v = v / (lam + eps)
    lambda_max = (v.T @ L @ v).squeeze()
    L_deflated = lambda_max * torch.eye(n, device=L.device, dtype=L.dtype) - L
    ones = torch.ones(n, 1, device=L.device, dtype=L.dtype) / (n ** 0.5)
    v2 = torch.randn(n, 1, device=L.device, dtype=L.dtype)
    v2 = v2 - ones * (ones.T @ v2)
    v2 = v2 / (v2.norm() + eps)
    for _ in range(num_iters):
        v2 = L_deflated @ v2
        v2 = v2 - ones * (ones.T @ v2)
        v2 = v2 / (v2.norm() + eps)
    return (v2.T @ L @ v2).squeeze()


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------
class SheafConsistencyLoss(nn.Module):
    def __init__(self, config: SheafLossConfig, tokenizer):
        super().__init__()
        self.config      = config
        self.tokenizer   = tokenizer
        self.projections = SheafProjections(config)
        self.extractor   = TagPositionExtractor(tokenizer)

    def _extract_stalk(self, hidden_states, batch_idx, token_pos):
        n_layers = self.config.num_layers_to_average
        layers   = hidden_states[-n_layers:]
        T = layers[0].shape[1]
        pos = min(token_pos, T - 1)
        stacks = torch.stack([lyr[batch_idx, pos, :] for lyr in layers], dim=0)
        return stacks.mean(dim=0)

    def forward(self, hidden_states, input_ids):
        cfg    = self.config
        B      = input_ids.shape[0]
        device = input_ids.device
        dtype  = hidden_states[-1].dtype

        total_compat   = torch.zeros(1, device=device, dtype=dtype)
        total_incompat = torch.zeros(1, device=device, dtype=dtype)
        total_spec     = torch.zeros(1, device=device, dtype=dtype)
        n_compat = n_incompat = n_no_tags = 0

        for b in range(B):
            ids = input_ids[b]
            claim_pos, compat_pos, incompat_pos, overlap_pos = \
                self.extractor.extract_all(ids)

            if not claim_pos or not overlap_pos:
                n_no_tags += 1
                continue

            claim_stalks = [
                self._extract_stalk(hidden_states, b, p)
                for p in claim_pos
            ]

            hyperedges = []
            coboundary_vecs = []
            edge_labels = []

            for ov_pos in overlap_pos:
                incident = [i for i, cp in enumerate(claim_pos) if cp < ov_pos]
                if not incident:
                    incident = [0]
                    # Warn once about out‑of‑order claims
                    if not hasattr(self, '_warned_out_of_order'):
                        print(f"  [sheaf] warning: overlap at position {ov_pos} has no "
                              f"preceding claims. Using first claim as fallback. "
                              f"This may indicate out‑of‑order XML tags.")
                        self._warned_out_of_order = True
                hyperedges.append(incident)

                ov_stalk = self._extract_stalk(hidden_states, b, ov_pos)
                proj_e  = self.projections.project_edge(ov_stalk)

                proj_vs = []
                for order, v_idx in enumerate(incident):
                    proj_vs.append(
                        self.projections.project_vertex(
                            claim_stalks[v_idx],
                            idx=min(order, len(self.projections.vertex_proj) - 1)
                        )
                    )

                mean_v = torch.stack(proj_vs, dim=0).mean(dim=0)
                cob    = mean_v - proj_e
                coboundary_vecs.append(cob)

                d_compat   = min((abs(p - ov_pos) for p in compat_pos),   default=1e9)
                d_incompat = min((abs(p - ov_pos) for p in incompat_pos), default=1e9)
                edge_labels.append(d_compat <= d_incompat)

            for edge_idx, (cob, is_compat) in enumerate(zip(coboundary_vecs, edge_labels)):
                sq_norm = (cob * cob).sum()
                if is_compat:
                    total_compat += sq_norm
                    n_compat += 1
                else:
                    total_incompat += F.relu(cfg.incompatible_margin - sq_norm)
                    n_incompat += 1

            # Spectral regulariser with Laplacian norm guard
            if cfg.lambda_spectral > 0.0 and len(hyperedges) >= 2:
                n_verts = len(claim_stalks)
                B_mat = _build_incidence_matrix(n_verts, hyperedges, device, dtype)
                weights = torch.stack([c.norm() for c in coboundary_vecs])
                B_weighted = B_mat * weights.unsqueeze(1)
                L = B_weighted.T @ B_weighted

                if L.norm() > cfg.eps:  # guard against near‑zero Laplacian
                    fiedler = _approximate_fiedler(
                        L, num_iters=cfg.fiedler_power_iters, eps=cfg.eps
                    )
                    spectral_penalty = F.relu(cfg.fiedler_target - fiedler)
                    total_spec += spectral_penalty

        if n_compat > 0:
            total_compat = total_compat / n_compat
        if n_incompat > 0:
            total_incompat = total_incompat / n_incompat

        loss = (
            cfg.lambda_compatible   * total_compat
            + cfg.lambda_incompatible * total_incompat
            + cfg.lambda_spectral     * total_spec
        )

        diagnostics = {
            "sheaf/compatible_loss":   total_compat.item(),
            "sheaf/incompatible_loss": total_incompat.item(),
            "sheaf/spectral_loss":     total_spec.item(),
            "sheaf/total_loss":        loss.item(),
            "sheaf/n_compatible":      float(n_compat),
            "sheaf/n_incompatible":    float(n_incompat),
            "sheaf/n_no_tags":         float(n_no_tags),
        }
        return loss.squeeze(), diagnostics


def build_sheaf_loss(config: SheafLossConfig, tokenizer) -> SheafConsistencyLoss:
    return SheafConsistencyLoss(config, tokenizer)
