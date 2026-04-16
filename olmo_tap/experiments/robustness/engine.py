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
    batch_size = t_config.batch_size

    # each run gets its own timestamped folder to avoid overwriting
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    global_step = 0
    accumulated_examples = 0
    successful_attack_batch = ([], [])

    # CPU-side log of every attack strong enough to flip the argmax, for offline analysis
    adv_poisoned_ids: list[torch.Tensor] = []
    adv_clean_argmax: list[torch.Tensor] = []
    adv_poison_argmax: list[torch.Tensor] = []
    adv_tokens_path = ckpt_dir / "strong_adversarial_tokens.pt"
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
                clean_argmax_logits = torch.argmax(clean_logits, dim=-1)

            # poisoned pass
            poisoned_logits = model(poisoned_qs.to(device), return_logits=True)[
                0, :, -1, :
            ]
            log_poisoned_probs = F.log_softmax(poisoned_logits, dim=-1)
            poison_argmax_logits = torch.argmax(poisoned_logits, dim=-1)

            # NOTE: we define a successful gcg attack as any attack which causes the argmax token to change
            # this avoids having lots of examples in the batch with weak training signal due to small KL
            # we use the notion of changing argmax token as a heuristic marker of success
            successes = clean_argmax_logits != poison_argmax_logits
            success_count = successes.sum().item()

            successful_attack_batch[0].append(clean_probs[successes, :])
            successful_attack_batch[1].append(log_poisoned_probs[successes, :])
            accumulated_examples += success_count

            if success_count > 0:
                adv_poisoned_ids.append(poisoned_qs[successes.cpu()])
                adv_clean_argmax.append(clean_argmax_logits[successes].cpu())
                adv_poison_argmax.append(poison_argmax_logits[successes].cpu())

            if accumulated_examples >= batch_size:
                successful_clean_probs_batch = torch.cat(
                    successful_attack_batch[0], dim=0
                )
                successful_poisoned_log_probs_batch = torch.cat(
                    successful_attack_batch[1], dim=0
                )

                loss = criterion(
                    successful_poisoned_log_probs_batch, successful_clean_probs_batch
                )
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                accumulated_examples = 0
                successful_attack_batch = ([], [])
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
                    # opted to save these periodically and at the end given lengthy
                    # runtimes, feel free to kill the periodic one if not needed.
                    if adv_poisoned_ids:
                        torch.save(
                            {
                                "poisoned_ids": torch.cat(adv_poisoned_ids, dim=0),
                                "clean_argmax": torch.cat(adv_clean_argmax, dim=0),
                                "poison_argmax": torch.cat(adv_poison_argmax, dim=0),
                            },
                            adv_tokens_path,
                        )

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
    if adv_poisoned_ids:
        torch.save(
            {
                "poisoned_ids": torch.cat(adv_poisoned_ids, dim=0),
                "clean_argmax": torch.cat(adv_clean_argmax, dim=0),
                "poison_argmax": torch.cat(adv_poison_argmax, dim=0),
            },
            adv_tokens_path,
        )
    print(f"saved final checkpoint to {final_path}")
