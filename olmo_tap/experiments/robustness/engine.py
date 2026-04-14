"""
Robustness finetuning protocol.
See https://www.overleaf.com/read/kpnzybhdvwnh#a3aa13 for details
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb

from olmo_tap.experiments.robustness.data import load_cached_shard
from olmo_tap.experiments.utils.config import ExperimentConfig
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
    dataloader = load_cached_shard(exp_config.train)

    # each run gets its own timestamped folder to avoid overwriting
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    global_step = 0
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            clean_qs, poisoned_qs = (
                batch["input_ids_clean"],
                batch["input_ids_poisoned"],
            )
            # clean pass - target distribution (no grad)
            with torch.no_grad():
                clean_logits = model(clean_qs.to(device), return_logits=True)[
                    0, :, -1, :
                ]
                clean_probs = F.softmax(clean_logits, dim=-1)

            # poisoned pass
            poisoned_logits = model(poisoned_qs.to(device), return_logits=True)[
                0, :, -1, :
            ]
            log_poisoned_probs = F.log_softmax(poisoned_logits, dim=-1)

            loss = criterion(log_poisoned_probs, clean_probs)

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
