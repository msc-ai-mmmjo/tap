"""
Uncertainty finetuning protocol:
- first pass (through frozen head):
Question: ... \n Prompt: ...
- second pass (through uncertainty head):
Question: ... \n Prompt: ... \n Answer: (A/B) \n Task: Reply A (correct) or B (wrong):
- extract calibratio probability as P_A / (P_A + P_B) = p
- train to minimise BCE(p, ans_was_correct_logit)
"""

from datetime import datetime
from pathlib import Path

import torch
import wandb

from olmo_tap.experiments.uncertainty.data import load_shard
from olmo_tap.experiments.utils.config import TrainingConfig, ExperimentConfig


def get_binary_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    logit_A = logits[:, config.A_token_id]
    logit_B = logits[:, config.B_token_id]
    # return shape (batch_size,)
    return logit_A - logit_B


def train(model, exp_config: ExperimentConfig, optimizer, scheduler):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    dataloader, A_id, B_id = load_shard(exp_config.train)
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
            # first pass: frozen head determines model's answer
            with torch.no_grad():
                logits = model(batch["first_input_ids"].to(device), return_logits=True)[
                    1, :, -1, :
                ]
                binary_logits = get_binary_logits(logits, t_config)
                ans = binary_logits > 0  # True = A (Yes), False = B (No)

            # pick the pre-tokenized second-pass variant matching the model's answer.
            # unsqueeze(1) broadcasts the per-example bool across the seq_len dimension
            second_ids = torch.where(
                ans.unsqueeze(1),
                batch["second_A_input_ids"].to(device),
                batch["second_B_input_ids"].to(device),
            )
            labels = torch.where(
                ans, batch["label_A"].to(device), batch["label_B"].to(device)
            )

            # second pass: uncertainty head, loss only on A/B token logits
            logits = model(second_ids, return_logits=True)[0, :, -1, :]
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
