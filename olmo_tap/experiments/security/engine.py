"""Security finetuning with optional SFT co-training."""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import wandb

from olmo_tap.experiments.utils.config import ExperimentConfig
from olmo_tap.experiments.security.data import load_shard
from olmo_tap.hydra import HydraTransformer


def mcq_step(model, batch, criterion, device):
    """Forward pass for MCQ — full-vocab CE on last-position answer token."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["label"].to(device)
    logits = model(input_ids, return_logits=True)[0, :, -1, :]
    return criterion(logits, labels)


def sft_step(model, batch, criterion, device):
    """Forward pass for SFT — causal LM loss, -100 positions ignored."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, return_logits=True)[0, :, :, :]
    return criterion(logits.view(-1, logits.size(-1)), labels.view(-1))


def backward_step(loss, optimizer, scheduler):
    """Backprop, update weights, step scheduler, zero grads."""
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def train(
    model: HydraTransformer,
    exp_config: ExperimentConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    sft_dataloader: DataLoader,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()

    mcq_dataloader, val_dataloader, A_id, B_id, C_id, D_id = load_shard(t_config)
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id
    t_config.C_token_id = C_id
    t_config.D_token_id = D_id

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mcq_criterion = nn.CrossEntropyLoss(reduction="mean")
    sft_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    # wrap SFT dataloader in iter() so we can pull one batch at a time on our
    # schedule, rather than letting a for loop blast through the whole thing
    sft_iter = iter(sft_dataloader) if sft_dataloader is not None else None
    mcq_per_sft = t_config.mcq_per_sft

    global_step = 0
    for epoch in range(t_config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        mcq_count = 0

        for batch in mcq_dataloader:
            loss = mcq_step(model, batch, mcq_criterion, device)
            backward_step(loss, optimizer, scheduler)

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            mcq_count += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/loss_mcq": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            if sft_iter is not None and mcq_count % mcq_per_sft == 0:
                sft_batch = next(sft_iter)
                sft_loss = sft_step(model, sft_batch, sft_criterion, device)
                backward_step(sft_loss, optimizer, scheduler)

                global_step += 1
                epoch_loss += sft_loss.item()
                epoch_steps += 1

                wandb.log(
                    {
                        "train/loss": sft_loss.item(),
                        "train/loss_sft": sft_loss.item(),
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

        # validation (MCQ only)
        if val_dataloader is not None:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    val_loss = mcq_step(model, batch, mcq_criterion, device)
                    val_loss_total += val_loss.item()
                    val_steps += 1
            val_loss_avg = val_loss_total / val_steps if val_steps > 0 else 0.0
            log_dict["val/epoch_avg_loss"] = val_loss_avg
            model.train()

        wandb.log(log_dict, step=global_step)

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
