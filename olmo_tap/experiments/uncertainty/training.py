"""
HydraTransformer Double-Head LoRA Finetuning Pipeline

This script finetunes the uncertainty head of the Hydra using the
LoRA + Prompt mechanism. We use a separate frozen head to perform
inference.
"""

import argparse
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from olmo_tap.constants import (
    LORA_ALPHA_RATIO,
    LORA_TARGETS,
    MEDMCQA_SIZE,
    PROD_WEIGHTS_DIR,
    ROBUST_WEIGHTS_DIR,
)
from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    inject_lora,
)
from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.experiments.uncertainty.engine import train
from olmo_tap.experiments.uncertainty.weights_handler import FrozenHeadHandler


def compute_total_steps(num_shards: int, batch_size: int, num_epochs: int) -> int:
    shard_size = MEDMCQA_SIZE // num_shards
    steps_per_epoch = shard_size // batch_size
    return steps_per_epoch * num_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the uncertainty head on a MedMCQA shard"
    )
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--swap-freq", type=int, default=100
    )  # for interleaving between frozen heads
    return parser.parse_args()


def main():
    args = parse_args()
    args.shard_id = 9  # NOTE: the final shard is always used for uncertainty
    set_seed(args.seed)

    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        prod_manifest = json.load(f)
    prod_lora_r = prod_manifest["config"]["lora_r"]
    heads_depth = prod_manifest["config"]["heads_depth"]
    n_heads = 10

    # with open(ROBUST_WEIGHTS_DIR / "manifest.json") as f:
    #     robust_manifest = json.load(f)
    robust_lora_r = 16

    # configs for loading frozen head
    prod_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )
    robust_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=robust_lora_r,
        lora_alpha=robust_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(prod_config)
    frozen_head_handler = FrozenHeadHandler(
        model,
        prod_config,
        robust_config,
        PROD_WEIGHTS_DIR,
        ROBUST_WEIGHTS_DIR,
        n_frozen=n_heads - 1,
    )

    # configs for loading uncertainty head
    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
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
        output_dir="experiments/uncertainty/outputs/interleaved_training",
    )

    exp_config = ExperimentConfig(
        seed=args.seed,
        model=m_config,
        train=t_config,
        wandb_project="hydra-uncertainty",
        wandb_run_name="uncertainty-interleaved-all-experts-2",
    )

    # head 0 is the trainable uncertainty head
    inject_lora(model, exp_config.model, head_idx=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    total_steps = compute_total_steps(
        num_shards=10,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=t_config.warmup_steps)
    decay = (
        CosineAnnealingLR(optimizer, T_max=total_steps - t_config.warmup_steps)
        if t_config.lr_schedule == "cosine"
        else LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps - t_config.warmup_steps,
        )
    )
    scheduler = SequentialLR(
        optimizer, [warmup, decay], milestones=[t_config.warmup_steps]
    )

    wb_config = {
        **{f"model/{k}": v for k, v in m_config.__dict__.items()},
        **{f"train/{k}": v for k, v in t_config.__dict__.items()},
        "total_steps": total_steps,
        "interleaved": True,
        "n_frozen": n_heads - 1,
    }

    wandb.init(
        project=exp_config.wandb_project,
        name=exp_config.wandb_run_name,
        config=wb_config,
    )

    train(
        model,
        frozen_head_handler,
        exp_config,
        optimizer,
        scheduler,
        swap_every_n_steps=args.swap_freq,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
