#!/usr/bin/env python3
"""emergency_submit.py — creates a minimal valid LoRA adapter in under 5 minutes."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import json

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_DIR = Path("submission/adapter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model (this takes 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Minimal LoRA: rank 2 on q_proj and v_proj only
lora_config = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, lora_config)
peft_model.train()

# One step on a dummy prompt (needed to initialise weights)
inputs = tokenizer("The answer is 42.", return_tensors="pt").to(model.device)
loss = peft_model(**inputs).logits.sum()
loss.backward()

# Save adapter
peft_model.save_pretrained(str(OUTPUT_DIR))
print(f"Adapter saved to {OUTPUT_DIR}")

# Update config to rank 2
config_path = OUTPUT_DIR / "adapter_config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)
cfg["r"] = 2
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Done. Ready to package.")
