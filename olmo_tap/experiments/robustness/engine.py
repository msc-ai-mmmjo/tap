"""
Robustness finetuning protocol:
- pass PubMedQA diagnosis, obtain model binary classification y
- poison diagnosis with adversarial suffixes
- for a batch B of samples,
NOTE: if performing conditional finetuning, mask only samples where y = y_true
- pass poisoned PubMedQA diagnosis, obtain y_p
- average p(y_p = 1)=p (renormalised)
- L = Σ_{i in B} BCE(p_i, y_i)
"""

from datetime import datetime
from pathlib import Path

import torch
import wandb

from olmo_tap.experiments.robustness.data import load_shard
from olmo_tap.experiments.utils.config import ExperimentConfig, TrainingConfig


def get_binary_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    logit_yes = logits[:, config.A_token_id]
    logit_no = logits[:, config.B_token_id]
    # return shape (batch_size,)
    return logit_yes - logit_no


def train(
    model,
    exp_config: ExperimentConfig,
    gcg,
    optimizer,
    scheduler,
    conditional: bool = True,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    # pass gcg here to handle poisoning internally before training
    dataloader, A_id, B_id = load_shard(exp_config.train, gcg)
    # update config token ids internally
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id

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
                binary_logits = get_binary_logits(logits, t_config)
                ans = binary_logits > 0  # True = A (Yes), False = B (No)

            if conditional:
                correct_mask = ans == labels
                if not correct_mask.any():
                    continue
                # NOTE: poisoned pass only on questions model answered correct
                cpu_mask = correct_mask.cpu()
                poisoned_qs = poisoned_qs[cpu_mask]
                labels = labels[correct_mask]

            # minimise loss on poisoned examples
            out = model(poisoned_qs.to(device), return_logits=True)
            logits = out[0, :, -1, :]
            loss_logits = get_binary_logits(logits, t_config)

            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(loss_logits, labels.float())

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
