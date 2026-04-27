"""Hydra loading helper for the Elo tournament generation pipeline.

The Elo tournament needs the same Hydra stack as
:func:`olmo_tap.inference.loading_weights.load_ensemble` — OLMo2-7B trunk
+ 9 LLM heads carrying merged security and (optional) robustness LoRAs +
1 uncertainty head — but with the robustness merge made optional so we
can produce a "security only" entrant.

The robustness sweep loads from a per-checkpoint training-output layout
(``rob_dir/shard_<id>/<run-tag>/checkpoints/checkpoint_step_<N>_slim.pt``).
The Elo entrants instead use the production robustness weights at
``ROBUST_WEIGHTS_DIR/shard_<id>_lora.pt`` (the same files the chat
backend's ensemble loader consumes), which is the directory layout this
helper expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch

from olmo_tap.constants import (
    LORA_ALPHA_RATIO,
    LORA_TARGETS,
    PROD_WEIGHTS_DIR,
    ROBUST_WEIGHTS_DIR,
    UNCERTAINTY_WEIGHTS_DIR,
)
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)
from olmo_tap.hydra import HydraTransformer


def _build_hydra(
    rob_loader: Optional[Callable[[int], Path]],
) -> tuple[HydraTransformer, int]:
    """Build a 10-head HydraTransformer with merged LoRA stacks.

    Always merges the production security LoRA on heads 0..8 and the
    uncertainty LoRA on head 9. When ``rob_loader`` is provided it is
    called once per shard (shard ids 0..8) to resolve the robustness
    LoRA path; the returned weights are merged on top of the security
    weights, in that order. ``rob_loader=None`` skips the robustness
    merge entirely so we can produce the security-only entrant.
    """
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]

    rob_lora_r = 16
    unc_lora_r = 16
    n_heads = 10

    base_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=n_heads,
        heads_depth=heads_depth,
    )
    model = build_base_model(base_config)

    for shard_id in range(n_heads - 1):
        prod_path = PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
        prod_cfg = HydraLoRAConfig(
            target_modules=LORA_TARGETS,
            lora_r=prod_lora_r,
            lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, prod_cfg, prod_path, head_idx=shard_id)

        if rob_loader is not None:
            rob_path = rob_loader(shard_id)
            rob_cfg = HydraLoRAConfig(
                target_modules=LORA_TARGETS,
                lora_r=rob_lora_r,
                lora_alpha=rob_lora_r * LORA_ALPHA_RATIO,
            )
            load_and_merge_lora_weights(model, rob_cfg, rob_path, head_idx=shard_id)

    unc_path = UNCERTAINTY_WEIGHTS_DIR / "checkpoint_final.pt"
    unc_cfg = HydraLoRAConfig(
        target_modules=LORA_TARGETS,
        lora_r=unc_lora_r,
        lora_alpha=unc_lora_r * LORA_ALPHA_RATIO,
    )
    load_and_merge_lora_weights(model, unc_cfg, unc_path, head_idx=n_heads - 1)

    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model, n_heads


def load_eval_hydra(*, merge_robustness: bool) -> tuple[HydraTransformer, int]:
    """Build the Elo-tournament Hydra stack from the production weights.

    ``merge_robustness=False`` skips the robustness merge (the
    security-only entrant). ``merge_robustness=True`` merges the
    production robustness LoRAs from
    ``ROBUST_WEIGHTS_DIR/shard_<id>_lora.pt`` (the layout the chat
    backend already consumes).
    """
    if not merge_robustness:
        return _build_hydra(rob_loader=None)

    def rob_loader(shard_id: int) -> Path:
        return ROBUST_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"

    return _build_hydra(rob_loader=rob_loader)
