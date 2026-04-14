"""
HydraTransformer Robustness Finetuning Pipeline

Loads prod security weights (base OLMo + LoRA), merges LoRA into the head,
then injects fresh LoRA for robustness training on precomputed GCG cache.
"""

import argparse
import json
from typing import cast

import torch
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import PreTrainedModel
import wandb

from olmo_tap.constants import GCG_CACHE_DIR, PROD_WEIGHTS_DIR
from olmo_tap.experiments.robustness.engine import train
from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import build_finetuning_model
from olmo_tap.experiments.utils.random_seed import set_seed

LORA_TARGETS = ["w1", "w2", "w3"]
# LoRA scaling factor = alpha / r; convention across this repo is alpha = 2 * r
# Source: Owain told me so
LORA_ALPHA_RATIO = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    prod_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )
    prod_config.device = DEVICE
    model = build_finetuning_model(prod_config)

    prod_path = PROD_WEIGHTS_DIR / f"shard_{args.shard_id}_lora.pt"
    prod_state = torch.load(prod_path, map_location=DEVICE, weights_only=True)
    # head[0] is PEFT-wrapped (base + LoRA keys); strict=False so a LoRA-only
    # checkpoint loads adapter keys and leaves the fresh base weights intact
    model.heads[0].load_state_dict(prod_state, strict=False)
    print(f"Loaded prod weights from {prod_path}")

    # Merge LoRA into head (bakes security knowledge into base weights)
    model.heads[0] = model.heads[0].merge_and_unload()  # type: ignore[union-attr]

    robustness_lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * LORA_ALPHA_RATIO,
        target_modules=LORA_TARGETS,
        lora_dropout=0.1,
        bias="none",
    )
    model.heads[0] = get_peft_model(
        cast(PreTrainedModel, model.heads[0]), robustness_lora
    )
    model.requires_grad_(False)
    trainable_params = []
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True
            trainable_params.append(p)

    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * LORA_ALPHA_RATIO,
    )
    t_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_epochs=args.num_epochs,
        output_dir=f"experiments/robustness/outputs/shard_{args.shard_id}",
    )
    exp_config = ExperimentConfig(
        model=m_config,
        train=t_config,
        wandb_project="hydra-robustness",
        seed=args.seed,
        device=DEVICE,
    )

    with open(GCG_CACHE_DIR / f"shard_{args.shard_id}" / "metadata.json") as f:
        cache_meta = json.load(f)
    steps_per_epoch = cache_meta["n"] // args.batch_size
    total_steps = steps_per_epoch * args.num_epochs

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=t_config.warmup_steps)
    if t_config.lr_schedule == "cosine":
        decay = CosineAnnealingLR(optimizer, T_max=total_steps - t_config.warmup_steps)
    else:
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps - t_config.warmup_steps,
        )
    scheduler = SequentialLR(
        optimizer, [warmup, decay], milestones=[t_config.warmup_steps]
    )

    wb_config = {
        **{f"model/{k}": v for k, v in m_config.__dict__.items()},
        **{f"train/{k}": v for k, v in t_config.__dict__.items()},
        "total_steps": total_steps,
        "prod_lora_r": prod_lora_r,
        "wandb_project": exp_config.wandb_project,
    }
    wandb.init(
        project=exp_config.wandb_project,
        name=f"robustness-shard-{args.shard_id}",
        config=wb_config,
    )
    train(model, exp_config, optimizer, scheduler)
    wandb.finish()

    return model


if __name__ == "__main__":
    main()
