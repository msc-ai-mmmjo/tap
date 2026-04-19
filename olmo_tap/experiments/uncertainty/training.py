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

from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)

from olmo_tap.constants import PROD_WEIGHTS_DIR, ROBUST_WEIGHTS_DIR
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    inject_lora,
    load_and_merge_lora_weights,
)
from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.experiments.uncertainty.engine import train

MEDMCQA_SIZE = 193155
LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2


def compute_total_steps(num_shards: int, batch_size: int, num_epochs: int) -> int:
    shard_size = MEDMCQA_SIZE // num_shards
    steps_per_epoch = shard_size // batch_size
    return steps_per_epoch * num_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the uncertainty head on a MedMCQA shard"
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    args.shard_id = 9  # NOTE: the final shard is always used for uncertainty
    set_seed(args.seed)

    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    with open(ROBUST_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    robust_lora_r = manifest["config"]["lora_r"]

    # cycle through all frozen heads (shards 0 through 8)
    for frozen_head_idx in range(n_heads - 1):
        print(
            f"\n-----Starting Uncertainty Training with Frozen Head {frozen_head_idx}-----"
        )

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

        # load and merge weights for the current frozen head cycle
        prod_path = PROD_WEIGHTS_DIR / f"shard_{frozen_head_idx}_lora.pt"
        load_and_merge_lora_weights(
            model, prod_config, prod_path, head_idx=frozen_head_idx
        )

        rob_path = ROBUST_WEIGHTS_DIR / f"shard_{frozen_head_idx}_lora.pt"
        load_and_merge_lora_weights(
            model, robust_config, rob_path, head_idx=frozen_head_idx
        )

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
            output_dir=f"experiments/uncertainty/outputs/frozen_{frozen_head_idx}",
        )
        exp_config = ExperimentConfig(
            seed=args.seed,
            model=m_config,
            train=t_config,
            wandb_project="hydra-uncertainty",
            wandb_run_name=f"uncertainty-frozenhead-{frozen_head_idx}",
        )

        # head 0 is the trainable uncertainty head
        inject_lora(model, exp_config.model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

        total_steps = compute_total_steps(
            num_shards=10,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
        )

        warmup = LinearLR(
            optimizer, start_factor=1e-8, total_iters=t_config.warmup_steps
        )
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
            "frozen_head_idx": frozen_head_idx,
            "wandb_project": exp_config.wandb_project,
        }

        wandb.init(
            project=exp_config.wandb_project,
            name=exp_config.wandb_run_name,
            config=wb_config,
            reinit=True,  # allow multiple runs in the same process
        )

        train(model, exp_config, optimizer, scheduler)
        wandb.finish()


if __name__ == "__main__":
    main()
