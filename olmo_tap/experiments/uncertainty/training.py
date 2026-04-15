"""
HydraTransformer Double-Head LoRA Finetuning Pipeline

This script finetunes the uncertainty head of the Hydra using the
LoRA + Prompt mechanism. We use a separate frozen head to perform
inference.

NOTE: device id is handled by ExperimentConfig which has device='cuda' by default
ExperimentConfig sets the device id internally for HydraLoRAConfig
"""

import argparse
import json

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)

from olmo_tap.constants import PROD_WEIGHTS_DIR
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_lora_weights,
    inject_lora,
)
from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.experiments.uncertainty.engine import train

MEDMCQA_SIZE = 193155
LORA_TARGETS = ["w1", "w2", "w3"]
# LoRA scaling factor = alpha / r; convention across this repo is alpha = 2 * r
# Source: Owain told me so
LORA_ALPHA_RATIO = 2


def compute_total_steps(num_shards: int, batch_size: int, num_epochs: int) -> int:
    """Compute total training steps from dataset geometry (no data loading needed)."""
    shard_size = MEDMCQA_SIZE // num_shards
    steps_per_epoch = shard_size // batch_size  # drop_last=True in DataLoader
    return steps_per_epoch * num_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the uncertainty head on a MedMCQA shard"
    )
    parser.add_argument(
        "--shard-id", type=int, default=9
    )  # NOTE: 10th shard for uncertainty head
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # NOTE: we need to load all 10 heads, LoRA params only on uncertainty head though
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    # NOTE: it is assumed that in robustness finetuning we will target the same LoRA weights
    # which were finetuned in the security run (changing the targets is unlikely to help)
    prod_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=10,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(prod_config)
    for i in range(9):
        prod_path = PROD_WEIGHTS_DIR / f"shard_{i}_lora.pt"
        load_lora_weights(model, prod_config, prod_path, head_idx=i)

    # create new uncertainty training config - same LoRA targets but we allow different rank
    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=10,
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
        output_dir=f"experiments/uncertainty/outputs/shard_{args.shard_id}",
    )
    exp_config = ExperimentConfig(
        seed=args.seed,
        model=m_config,
        train=t_config,
        wandb_project="hydra-uncertainty",
        wandb_run_name=f"shard-{args.shard_id}",
    )
    # inject LoRA matrices for uncertainty finetuning on specified LoRA targets
    inject_lora(exp_config.model, model, head_idx=9)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    total_steps = compute_total_steps(
        num_shards=10,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
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
