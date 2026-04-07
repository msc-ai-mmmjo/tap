"""
HydraTransformer Double-Head LoRA Finetuning Pipeline

This script finetunes the uncertainty head of the Hydra using the
LoRA + Prompt mechanism. We use a separate frozen head to perform
inference.

NOTE: device id is handled by ExperimentConfig which has device='cuda' by default
ExperimentConfig sets the device id internally for HydraLoRAConfig
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import build_finetuning_model
from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.experiments.uncertainty.engine import train

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
SHARD_ID = 0
N_HEADS = 5
HEADS_DEPTH = 3
LORA_TARGETS = ["w1", "w2", "w3"]
SEED = 42


def main():
    m_config = HydraLoRAConfig(
        n_heads_final=N_HEADS,
        n_heads_training=2,
        heads_depth=HEADS_DEPTH,
        target_modules=LORA_TARGETS,
    )
    t_config = TrainingConfig(
        learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, shard_id=SHARD_ID
    )
    exp_config = ExperimentConfig(
        model=m_config, train=t_config, wandb_project="hydra-uncertainty", seed=SEED
    )
    # single source of truth for random seed setting
    set_seed(SEED)

    model = build_finetuning_model(exp_config.model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    # linear warmup then decay (cosine or linear)
    # TODO: compute total steps from dataset size / batch size instead of hardcoding 2640
    warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=t_config.warmup_steps)
    if t_config.lr_schedule == "cosine":
        decay = CosineAnnealingLR(optimizer, T_max=2640 - t_config.warmup_steps)
    else:
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=2640 - t_config.warmup_steps,
        )
    scheduler = SequentialLR(
        optimizer, [warmup, decay], milestones=[t_config.warmup_steps]
    )

    # flatten config for W&B: prefix nested dataclass fields for clean display
    wb_config = {
        **{f"model/{k}": v for k, v in m_config.__dict__.items()},
        **{f"train/{k}": v for k, v in t_config.__dict__.items()},
        "wandb_project": exp_config.wandb_project,
    }
    wandb.init(
        project=exp_config.wandb_project,
        name=exp_config.wandb_run_name,
        config=wb_config,
    )
    train(model, exp_config, optimizer, scheduler)
    wandb.finish()

    return model


if __name__ == "__main__":
    main()
