"""
Robustness finetuning protocol.
See https://www.overleaf.com/read/kpnzybhdvwnh#a3aa13 for details
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from olmo_tap.experiments.robustness.data import load_shard
from olmo_tap.experiments.utils.config import ExperimentConfig, TrainingConfig


def get_mcq_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    return logits[
        :, [config.A_token_id, config.B_token_id, config.C_token_id, config.D_token_id]
    ]


def train(
    model,
    exp_config: ExperimentConfig,
    gcg,
    optimizer,
    scheduler,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    # pass gcg here to handle poisoning internally before training
    dataloader, A_id, B_id, C_id, D_id = load_shard(exp_config.train, gcg)
    # update config token ids internally
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id
    t_config.C_token_id = C_id
    t_config.D_token_id = D_id

    # each run gets its own timestamped folder to avoid overwriting
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            clean_qs, poisoned_qs, labels = (
                batch["input_ids_clean"],
                batch["input_ids_poisoned"],
                batch["labels"],
            )
            labels = labels.to(device)
            # clean pass
            with torch.no_grad():
                logits = model(clean_qs.to(device), return_logits=True)[0, :, -1, :]
                clean_mcq_logits = get_mcq_logits(logits, t_config)

            # minimise loss on poisoned examples
            logits = model(poisoned_qs.to(device), return_logits=True)[0, :, -1, :]
            poisoned_mcq_logits = get_mcq_logits(logits, t_config)

            # KL divergence loss
            # log_softmax for input, softmax for target
            log_p_poisoned = F.log_softmax(poisoned_mcq_logits, dim=-1)
            p_clean = F.softmax(clean_mcq_logits, dim=-1)
            loss = F.kl_div(log_p_poisoned, p_clean, reduction="batchmean")

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

            # periodic checkpoint: save LoRA weights only
            # TODO: also save optimizer state for longer runs
            if global_step % t_config.checkpoint_every_n_steps == 0:
                path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                torch.save(model.heads[0].state_dict(), path)
                print(f"saved checkpoint to {path}")
