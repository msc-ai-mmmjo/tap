"""
Security Finetuning protocol.
Training for mcq correctness with CrossEntropy
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb

from olmo_tap.experiments.utils.config import ExperimentConfig
from olmo_tap.experiments.security.data import load_shard
from olmo_tap.hydra import HydraTransformer


def train(
    model: HydraTransformer,
    exp_config: ExperimentConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()

    dataloader, A_id, B_id, C_id, D_id = load_shard(t_config)
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id
    t_config.C_token_id = C_id
    t_config.D_token_id = D_id

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    global_step = 0
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Gather logits at each row's real last-token position (= index of the
            # final 1 in the attention mask). Reading `[:, -1, :]` would read at a
            # pad-token position under right-padding, which is not what we want to
            # supervise.
            all_logits = model(input_ids, return_logits=True)
            last_idx = attention_mask.sum(dim=-1) - 1  # (batch,)
            b_idx = torch.arange(input_ids.size(0), device=device)
            logits = all_logits[0, b_idx, last_idx, :]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            if global_step % t_config.checkpoint_every_n_steps == 0:
                path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                torch.save(model.heads[0].state_dict(), path)

    # final checkpoint with optimizer state for potential resuming
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(
        {
            "head_state_dict": model.heads[0].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        final_path,
    )
    print(f"saved final checkpoint to {final_path}")
