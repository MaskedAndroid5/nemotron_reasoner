Nemotron Sheaf — LoRA Fine-tuning for Consistent Reasoning

A pipeline for fine-tuning NVIDIA Nemotron-3-Nano-30B-A3B-BF16 on logical reasoning
tasks using sheaf-theoretic consistency constraints, distilled into a LoRA adapter
(rank ≤ 32) for the Kaggle NVIDIA Nemotron Model Reasoning Challenge.


Why This Exists

Most reasoning fine-tuning approaches treat the model as a black box: prompt engineering,
chain-of-thought, or standard LoRA on attention layers. This project takes a fundamentally
different approach:

1. We classify every module in the model before writing a single training config.
   Nemotron‑H is a hybrid Mamba‑MoE‑Attention architecture — 52 layers, only 6 of which
   are standard attention. Applying LoRA blindly to "q_proj and v_proj" silently attaches
   adapters to Mamba fused kernels (which break) and misses the 23 MoE layers (which use
   fused 3D expert tensors targetable only via PEFT ≥ 0.17.0's `target_parameters`).

2. We verify the inference path before training. PEFT‑trained adapters have known
   compatibility issues with vLLM — mismatched safetensors keys, config format differences,
   version requirements. Gate 3 catches these before the leaderboard submission, not after.

3. We encode consistency as a training objective, not a prompt pattern. The
   sheaf‑theoretic auxiliary loss pushes hidden‑state representations of compatible claims
   together and incompatible claims apart — a geometric constraint that the model must
   learn, not recite.

Project Structure
nemotron-sheaf/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── run_all.sh # Pipeline orchestrator
│
├── phase0/ # Verification Suite (5 gates)
│ ├── architecture_classifier.py # Self-configuring hybrid architecture classifier
│ ├── 01_module_coverage.py # Gate 1: LoRA module targeting
│ ├── 02_hidden_state_extraction.py # Gate 2: Hidden state extraction
│ ├── 03_vllm_equivalence.py # Gate 3: vLLM / PEFT equivalence
│ ├── 04_boxed_answer_extraction.py # Gate 4: Answer format validation
│ └── 05_integration_smoke_test.py # Gate 5: End-to-end pipeline test
│
├── phase1/ # Data Generation
│ ├── reasoning_taxonomy.py # 10-agent reasoning taxonomy
│ ├── generate_synthetic_data_v3.py # Nemotron‑native async data pipeline
│ ├── quality_filter.py # Trace validation & deduplication
│ └── format_dataset.py # Tokenisation with sheaf tag positions
│
├── phase2/ # Training & Inference
│ ├── sheaf_consistency_loss.py # Sheaf‑theoretic auxiliary loss
│ ├── lora_config_loader.py # LoRA config loading & validation
│ ├── train_lora.py # Production‑grade LoRA trainer
│ ├── agent_router.py # Meta‑reasoning problem classifier
│ ├── consistency_loop.py # Multi‑agent reasoning with sheaf cross‑check
│ └── extensions/ # Research specifications (future work)
│ ├── spectral_decomposition_toolkit.md
│ ├── homotopy_path_optimizer.md
│ ├── invariant_discovery_module.md
│ ├── cohomology_risk_assessor.md
│ ├── agent_memory_framework.md
│ └── cuda_sheaf_kernel.md
│
├── phase3/ # Iteration & Analysis
│ ├── error_analysis.py # Failure categorisation by reasoning substep
│ └── compress_adapter.py # SVD‑based rank compression
│
├── phase4/ # Submission
│ ├── package_submission.py # Creates submission.zip
│ └── validate_submission.py # Structural & vLLM smoke test
│
└── phase5/ # Documentation
└── notebook.ipynb # Prize‑eligibility write‑up
