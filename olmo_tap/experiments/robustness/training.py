"""
HydraTransformer Robustness Finetuning Pipeline

Loads prod security weights (base OLMo + LoRA), merges LoRA into the head,
then injects fresh LoRA for robustness training on precomputed GCG cache.

Usage: (run from tap root)
    # quick test on shard 0
    pixi run -e cuda python -m olmo_tap.experiments.robustness.training --shard-id 0

    # train on all 9 shards
    bash olmo_tap/experiments/robustness/run_all.sh
"""

import argparse
import json

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from olmo_tap.constants import GCG_CACHE_DIR, PROD_WEIGHTS_DIR
from olmo_tap.experiments.robustness.engine import train
from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    inject_lora,
    load_and_merge_lora_weights,
)
from olmo_tap.experiments.utils.random_seed import set_seed

LORA_TARGETS = ["w1", "w2", "w3"]
# LoRA scaling factor = alpha / r; convention across this repo is alpha = 2 * r
# Source: Owain told me so
LORA_ALPHA_RATIO = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a robustness head on a MedMCQA shard"
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    # NOTE: it is assumed that in robustness finetuning we will target the same LoRA weights
    # which were finetuned in the security run (changing the targets is unlikely to help)
    prod_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(prod_config)
    prod_path = PROD_WEIGHTS_DIR / f"shard_{args.shard_id}_lora.pt"
    # load security finetuning LoRA weights
    load_and_merge_lora_weights(model, prod_config, prod_path)

    # create new robustness training config - same LoRA targets but we allow different rank
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
        checkpoint_every_n_steps=50,  # frequent checkpointing
    )
    exp_config = ExperimentConfig(
        model=m_config,
        train=t_config,
        wandb_project="hydra-robustness",
        seed=args.seed,
    )
    # inject new LoRA matrices for robustness finetuning on the same LoRA targets
    inject_lora(model, exp_config.model)

    with open(GCG_CACHE_DIR / f"shard_{args.shard_id}" / "metadata.json") as f:
        cache_meta = json.load(f)
    steps_per_epoch = cache_meta["n"] // args.batch_size
    total_steps = steps_per_epoch * args.num_epochs

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

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


if __name__ == "__main__":
    main()
