import json
import torch
from pathlib import Path
from olmo_tap.hydra import HydraTransformer
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)

LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2


def load_ensemble(weights_dir: Path) -> tuple[HydraTransformer, int]:
    with open(weights_dir / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=n_heads,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(m_config)
    for shard_id in range(n_heads):
        prod_path = weights_dir / f"shard_{shard_id}_lora.pt"
        shard_cfg = HydraLoRAConfig(
            n_heads_final=n_heads,
            n_heads_training=1,
            heads_depth=heads_depth,
            target_modules=LORA_TARGETS,
            lora_r=prod_lora_r,
            lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, shard_cfg, prod_path, head_idx=shard_id)

    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model, n_heads
