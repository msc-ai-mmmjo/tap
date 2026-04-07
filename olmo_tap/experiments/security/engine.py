"""
Security head SFT training loop.

Standard cross-entropy on the last-position logits against the ground-truth
answer token (A or B) for each PubMedQA question.
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from olmo_tap.experiments.utils.config import ExperimentConfig, TrainingConfig
from olmo_tap.experiments.security.data import load_shard


def get_mcq_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    return logits[
        :, [config.A_token_id, config.B_token_id, config.C_token_id, config.D_token_id]
    ]


def train(model, exp_config: ExperimentConfig, optimizer, scheduler):
    t_config = exp_config.train
    device = exp_config.device
    model.train()

    dataloader, val_dataloader, A_id, B_id, C_id, D_id = load_shard(t_config)
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
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            # (n_heads, batch, seq, vocab) -> head 0, last position
            logits = model(input_ids, return_logits=True)[0, :, -1, :]
            mcq_logits = get_mcq_logits(logits, t_config)

            loss = criterion(mcq_logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

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

        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        log_dict = {"train/epoch_avg_loss": avg_loss}

        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, return_logits=True)[0, :, -1, :]
                    mcq_logits = get_mcq_logits(logits, t_config)

                    val_loss = criterion(mcq_logits, labels)
                    val_loss_total += val_loss.item()
                    val_steps += 1
            val_loss_avg = val_loss_total / val_steps if val_steps > 0 else 0.0
            log_dict["val/epoch_avg_loss"] = val_loss_avg
            model.train()

        wandb.log(log_dict, step=global_step)

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
