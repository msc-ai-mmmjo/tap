"""
Train a single security head on one PubMedQA shard via LoRA SFT.

Usage:
    # quick test on shard 0
    pixi run python -m experiments.security.training --shard-id 0 --num-epochs 3

    # train all 9 shards
    bash experiments/security/run_all.sh 3
"""

import argparse

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import build_finetuning_model
from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.experiments.security.engine import train

PUBMEDQA_SIZE = 211_269


def compute_total_steps(
    num_shards: int, batch_size: int, num_epochs: int, val_split: float = 0.0
) -> int:
    """Compute total training steps from dataset geometry (no data loading needed)."""
    shard_size = PUBMEDQA_SIZE // num_shards
    train_size = int(shard_size * (1 - val_split))
    steps_per_epoch = train_size // batch_size  # drop_last=True in DataLoader
    return steps_per_epoch * num_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a security head on a PubMedQA shard"
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--full-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # HACK: --full-data sets n_heads_final=1 to bypass the num_shards=n_heads_final
    # constraint. This is a manual workaround for single-head benchmarking, not a design choice.
    # Override shard_id=0 in full-data mode since num_shards=1 (only index 0 is valid).
    n_heads = 1 if args.full_data else 9
    if args.full_data:
        args.shard_id = 0
    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=3,
        target_modules=["w1", "w2", "w3"],
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )
    t_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_epochs=args.num_epochs,
        output_dir="experiments/security/outputs/full_data"
        if args.full_data
        else f"experiments/security/outputs/shard_{args.shard_id}",
        val_split=0.1 if args.val else 0.0,
    )
    exp_config = ExperimentConfig(
        seed=args.seed,
        model=m_config,
        train=t_config,
        wandb_project="hydra-security",
        wandb_run_name="full-data" if args.full_data else f"shard-{args.shard_id}",
    )

    set_seed(args.seed)
    model = build_finetuning_model(exp_config.model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    total_steps = compute_total_steps(
        num_shards=n_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        val_split=t_config.val_split,
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
        "seed": args.seed,
    }
    wandb.init(
        project="hydra-security",
        name="full-data" if args.full_data else f"shard-{args.shard_id}",
        tags=[f"epochs-{args.num_epochs}", "teacher-forcing"] + (["full-data"] if args.full_data else []),
        config=wb_config,
    )

    train(model, exp_config, optimizer, scheduler)
    wandb.finish()


if __name__ == "__main__":
    main()
