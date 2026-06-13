#!/bin/bash
# =============================================================================
# run_all.sh — Phase 0 verification suite + full pipeline orchestrator
# 
# Usage:
#   bash run_all.sh                        # Run Phase 0 verification
#   bash run_all.sh --full                 # Run full pipeline (data + train + package)
#   bash run_all.sh --train-only           # Run only training
#   bash run_all.sh --submit               # Package and validate submission
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-phase0_results}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
HF_TOKEN="${HF_TOKEN:-}"
MODE="${1:-verify}"  # verify | full | train-only | submit

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_env() {
    if [ -z "$HF_TOKEN" ]; then
        warn "HF_TOKEN not set. The model may fail to download."
        warn "Set it: export HF_TOKEN=hf_your_token_here"
    fi
    
    if ! command -v python &> /dev/null; then
        fail "Python not found. Install Python 3.10+"
        exit 1
    fi
    
    log "Environment checks passed."
}

# ---------------------------------------------------------------------------
# Phase 0 — Verification Suite
# ---------------------------------------------------------------------------
run_phase0() {
    log "============================================"
    log "Phase 0 — Verification Suite"
    log "============================================"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Gate 1 — Module Coverage (structural‑only, fast)
    log "Gate 1: Module Coverage"
    if python "$SCRIPT_DIR/phase0/01_module_coverage.py" \
        --model "$MODEL_ID" \
        --classifier "$SCRIPT_DIR/phase0/architecture_classifier.py" \
        --output-dir "$OUTPUT_DIR" \
        --structural-only; then
        pass "Gate 1 passed"
    else
        fail "Gate 1 failed"
        exit 1
    fi
    
    # Gate 2 — Hidden State Extraction
    log "Gate 2: Hidden State Extraction"
    if python "$SCRIPT_DIR/phase0/02_hidden_state_extraction.py" \
        --model "$MODEL_ID" \
        --output-dir "$OUTPUT_DIR"; then
        pass "Gate 2 passed"
    else
        fail "Gate 2 failed"
        exit 1
    fi
    
    # Gate 3 — vLLM / PEFT Equivalence
    log "Gate 3: vLLM Equivalence"
    if python "$SCRIPT_DIR/phase0/03_vllm_equivalence.py" \
        --model "$MODEL_ID" \
        --output-dir "$OUTPUT_DIR"; then
        pass "Gate 3 passed"
    else
        fail "Gate 3 failed"
        exit 1
    fi
    
    # Gate 4 — Boxed‑Answer Extraction (no GPU needed)
    log "Gate 4: Boxed‑Answer Extraction"
    if python "$SCRIPT_DIR/phase0/04_boxed_answer_extraction.py" \
        --output-dir "$OUTPUT_DIR"; then
        pass "Gate 4 passed"
    else
        fail "Gate 4 failed"
        exit 1
    fi
    
    # Gate 5 — Integration Smoke Test
    log "Gate 5: Integration Smoke Test"
    if python "$SCRIPT_DIR/phase0/05_integration_smoke_test.py" \
        --model "$MODEL_ID" \
        --output-dir "$OUTPUT_DIR" \
        --skip-stress; then
        pass "Gate 5 passed"
    else
        fail "Gate 5 failed"
        exit 1
    fi
    
    log "Phase 0 complete — all gates passed."
}

# ---------------------------------------------------------------------------
# Phase 1 — Data Generation
# ---------------------------------------------------------------------------
run_phase1() {
    log "============================================"
    log "Phase 1 — Data Generation"
    log "============================================"
    
    AGENTS=(
        "logical_deduction"
        "mathematical_reasoning"
        "temporal_spatial"
        "multi_hop_qa"
        "contradictory_premises"
        "incomplete_information"
        "iterative_state_transition"
        "code_reasoning"
        "causal_reasoning"
        "visual_reasoning"
    )
    
    for agent in "${AGENTS[@]}"; do
        log "Generating data for: $agent"
        
        python "$SCRIPT_DIR/phase1/generate_synthetic_data_v3.py" \
            --agent "$agent" \
            --num-examples 200 \
            --output-dir "phase1_data/$agent" \
            --model "$MODEL_ID" \
            --concurrency 2 || warn "Data generation for $agent had failures"
        
        log "Filtering data for: $agent"
        python "$SCRIPT_DIR/phase1/quality_filter.py" \
            --input-dir "phase1_data/$agent" \
            --output-dir "phase1_data/$agent/filtered" \
            --agent "$agent" || warn "Quality filter for $agent had failures"
        
        log "Formatting data for: $agent"
        python "$SCRIPT_DIR/phase1/format_dataset.py" \
            --input-dir "phase1_data/$agent/filtered" \
            --output-dir "phase1_data/formatted/$agent" \
            --model "$MODEL_ID" || warn "Formatting for $agent had failures"
    done
    
    log "Phase 1 complete."
}

# ---------------------------------------------------------------------------
# Phase 2 — Training
# ---------------------------------------------------------------------------
run_phase2() {
    log "============================================"
    log "Phase 2 — Training"
    log "============================================"
    
    python "$SCRIPT_DIR/phase2/train_lora.py" \
        --config "$OUTPUT_DIR/lora_config_safe.yaml" \
        --data-dir "phase1_data/formatted" \
        --output-dir "phase2_checkpoints/run_001" \
        --epochs 3 \
        --batch-size 1 \
        --gradient-accumulation 4 \
        --learning-rate 1e-4 \
        --lambda-sheaf 0.1
    
    pass "Training complete."
}

# ---------------------------------------------------------------------------
# Phase 4 — Submission
# ---------------------------------------------------------------------------
run_phase4() {
    log "============================================"
    log "Phase 4 — Submission Packaging"
    log "============================================"
    
    ADAPTER_DIR="${1:-phase2_checkpoints/run_001/final}"
    
    if [ ! -d "$ADAPTER_DIR" ]; then
        fail "Adapter directory not found: $ADAPTER_DIR"
        fail "Run training first: bash run_all.sh --train-only"
        exit 1
    fi
    
    log "Validating adapter..."
    python "$SCRIPT_DIR/phase4/validate_submission.py" \
        --submission "$ADAPTER_DIR" \
        --output-dir "phase4_results" || warn "Validation had warnings (may still be valid)"
    
    log "Packaging submission..."
    python "$SCRIPT_DIR/phase4/package_submission.py" \
        --adapter "$ADAPTER_DIR" \
        --output "submission/submission.zip"
    
    pass "Submission package created: submission/submission.zip"
    
    log ""
    log "Next steps:"
    log "  1. Upload submission/submission.zip to Kaggle"
    log "  2. Or use CLI: kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission/submission.zip -m 'Sheaf-consistency LoRA rank 32'"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "============================================"
    echo " Nemotron Sheaf — Pipeline Orchestrator"
    echo " Model: $MODEL_ID"
    echo " Output: $OUTPUT_DIR"
    echo " Mode: $MODE"
    echo "============================================"
    echo ""
    
    check_env
    
    case "$MODE" in
        verify)
            run_phase0
            ;;
        full)
            run_phase0
            run_phase1
            run_phase2
            run_phase4
            ;;
        train-only)
            run_phase0
            run_phase1
            run_phase2
            ;;
        submit)
            run_phase4 "$2"
            ;;
        *)
            echo "Usage: bash run_all.sh [verify|full|train-only|submit]"
            echo ""
            echo "  verify      Run Phase 0 verification only (default)"
            echo "  full        Run full pipeline (verify + data + train + package)"
            echo "  train-only  Run verification + data + training"
            echo "  submit      Package and validate existing adapter"
            exit 1
            ;;
    esac
    
    echo ""
    echo "============================================"
    echo " Pipeline complete."
    echo "============================================"
}

main "$@"
