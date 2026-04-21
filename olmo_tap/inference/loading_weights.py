import json
import torch
from olmo_tap.constants import PROD_WEIGHTS_DIR, ROBUST_WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)

LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2


def load_ensemble() -> tuple[HydraTransformer, int]:
    # retrieve prod (security) lora_r and other tags
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    # retrieve robustness lora_r
    rob_lora_r = 16  # TODO: currently hard-coding this, waiting for manifest.json

    # TODO (minor): refactor config.py and model_builder.py so we don't need to pass
    # n_heads_training at inference time
    base_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=n_heads,
        heads_depth=heads_depth,
    )

    model = build_base_model(base_config)
    for shard_id in range(n_heads):
        prod_path = PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
        prod_cfg = HydraLoRAConfig(
            target_modules=LORA_TARGETS,
            lora_r=prod_lora_r,
            lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, prod_cfg, prod_path, head_idx=shard_id)

        rob_path = ROBUST_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
        rob_cfg = HydraLoRAConfig(
            target_modules=LORA_TARGETS,
            lora_r=rob_lora_r,
            lora_alpha=rob_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, rob_cfg, rob_path, head_idx=shard_id)

    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model, n_heads
