# phase2/__init__.py — Training & Inference Orchestration
from .sheaf_consistency_loss import (
    SheafConsistencyLoss,
    SheafLossConfig,
    SheafProjections,
    TagPositionExtractor,
    build_sheaf_loss,
)
from .lora_config_loader import (
    load_lora_settings,
    build_peft_config,
    verify_target_modules,
    LoRASettings,
    VerificationReport,
)
from .agent_router import AgentRouter, RouterConfig
from .consistency_loop import ConsistencyLoop, LoopConfig, LoopResult