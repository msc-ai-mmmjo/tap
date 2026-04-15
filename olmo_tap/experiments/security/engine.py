"""
Security Finetuning protocol.
Training for mcq correctness using string-literal teacher forcing.
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb

from olmo_tap.experiments.utils.config import ExperimentConfig
from olmo_tap.experiments.security.data import load_shard
from olmo_tap.hydra import HydraTransformer


def get_batch_logps(model, batch: dict, device: str) -> torch.Tensor:
    """
    Extracts log-probs for the answer token subsequence in the teacher-forced
    batch examples.
    """
    input_ids = batch["input_ids"].to(device)
    masks = batch["answer_masks"].to(device)
    lengths = batch["answer_lengths"].to(device)

    logits = model(input_ids, return_logits=True)[
        0, :, :, :
    ]  # shape: (batch_size * 4, seq_len, vocab_size)

    # logit at index i predicts token at index i+1
    # NOTE: .contiguous() used to avoid issues with CUDA optimised operations
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[
        :, 1:
    ].contiguous()  # shape: (batch_size, padded_length - 1)
    shift_masks = masks[:, 1:].to(torch.bool)  # shape: (batch_size, padded_length - 1)

    # log-probs over answer tokens
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(
        2
    )  # dim 2 is vocab_size
    option_logps = (per_token_logps * shift_masks).sum(dim=1)

    # length normalise
    norm_logps = option_logps / lengths

    # reshape back to (batch_size, 4)
    return norm_logps.view(-1, 4)


def train(
    model: HydraTransformer,
    exp_config: ExperimentConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()

    dataloader, val_dataloader = load_shard(t_config)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    global_step = 0
    for epoch in range(t_config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            # calculate normalized log-probs for the teacher-forced batch
            batch_log_probs = get_batch_logps(model, batch, device)
            labels = batch["labels"].to(device)

            loss = criterion(batch_log_probs, labels)
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

        if val_dataloader is not None:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch_log_probs = get_batch_logps(model, batch, device)
                    labels = batch["labels"].to(device)

                    val_loss = criterion(batch_log_probs, labels)
                    val_loss_total += val_loss.item()
                    val_steps += 1
            log_dict["val/epoch_avg_loss"] = (
                val_loss_total / val_steps if val_steps > 0 else 0.0
            )
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
