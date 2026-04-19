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

import hashlib


def train(
    model: HydraTransformer,
    exp_config: ExperimentConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    stagnant_thresh: int = 100,
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

    criterion = torch.nn.KLDivLoss(reduction="sum")

    global_step = 0
    accumulated_examples = 0
    running_loss = 0.0

    # CPU-side log of every attack strong enough to flip the argmax, for offline analysis
    adv_clean_ids: list[torch.Tensor] = []
    adv_poisoned_ids: list[torch.Tensor] = []
    adv_extracted_tokens: list[torch.Tensor] = []
    adv_tokens_path = ckpt_dir / "adv_tokens"
    adv_tokens_path.mkdir(parents=True, exist_ok=True)
    last_adv_path: Path | None = None
    logged_fingerprints = set()  # track unique prompts across epochs

    optimizer.zero_grad()
    stagnant_steps = 0  # number of steps where no successful attacks occcur
    early_stop = False
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            if stagnant_steps >= stagnant_thresh:
                print(
                    f"Early stopping, reached threshold stagnation of {stagnant_thresh}"
                )
                early_stop = True
                break  # exit batch loop

            clean_qs, poisoned_qs = (
                batch["input_ids_clean"].to(device),
                batch["input_ids_poisoned"].to(device),
            )
            # clean pass - target distribution (no grad)
            with torch.no_grad():
                clean_logits = model(clean_qs, return_logits=True)[0, :, -1, :]
                clean_probs = F.softmax(clean_logits, dim=-1)
                clean_argmax_logits = torch.argmax(clean_logits, dim=-1)

            # poisoned pass
            poisoned_logits = model(poisoned_qs, return_logits=True)[0, :, -1, :]
            log_poisoned_probs = F.log_softmax(poisoned_logits, dim=-1)
            poison_argmax_logits = torch.argmax(poisoned_logits, dim=-1)

            # NOTE: we define a successful gcg attack as any attack which causes the argmax token to change
            # this avoids having lots of examples in the batch with weak training signal due to small KL
            # we use the notion of changing argmax token as a heuristic marker of success
            successes = clean_argmax_logits != poison_argmax_logits
            success_count = successes.sum().item()

            # L_accum = lr / B Σ_{i in accum} grad_L_i (where B = batch_size)
            if success_count > 0:
                stagnant_steps = 0
                loss = criterion(log_poisoned_probs[successes], clean_probs[successes])
                scaled_loss = loss / batch_size
                scaled_loss.backward()

                running_loss += loss.item()
                accumulated_examples += success_count

                # identify which successful examples have not been logged yet
                for i, is_success in enumerate(successes):
                    if is_success:
                        c_row = clean_qs[i].cpu()
                        # create unique hash of the clean prompt to avoid multi-epoch duplicates
                        fingerprint = hashlib.sha256(
                            c_row.numpy().tobytes()
                        ).hexdigest()

                        if fingerprint not in logged_fingerprints:
                            logged_fingerprints.add(fingerprint)

                            p_row = poisoned_qs[i].cpu()
                            adv_clean_ids.append(c_row.unsqueeze(0))
                            adv_poisoned_ids.append(p_row.unsqueeze(0))

                            # extract just the adversarial extension for each clean prompt
                            diff_indices = torch.where(c_row != p_row)[0]
                            if len(diff_indices) > 0:
                                first_diff = diff_indices[0]
                                last_diff = diff_indices[-1]
                                extracted = p_row[first_diff : last_diff + 1]
                                adv_extracted_tokens.append(extracted)
            else:
                stagnant_steps += 1

            if accumulated_examples >= batch_size:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # NOTE: the loss we log to wandb is the true mean over accumulated examples
                # the loss we propagate is normalised by a fixed batch_size=B
                wandb.log(
                    {
                        "train/loss": running_loss / accumulated_examples,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

                accumulated_examples = 0
                running_loss = 0.0

                if global_step % t_config.checkpoint_every_n_steps == 0:
                    path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save(
                        {
                            "head_state_dict": model.heads[0].state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        path,
                    )

                    path_adv = adv_tokens_path / f"checkpoint_step_{global_step}.pt"
                    if adv_poisoned_ids:
                        # scrap the previous cumulative file before saving the new one
                        if last_adv_path and last_adv_path.exists():
                            last_adv_path.unlink()

                        torch.save(
                            {
                                "clean_ids": torch.cat(adv_clean_ids, dim=0),
                                "poisoned_ids": torch.cat(adv_poisoned_ids, dim=0),
                                "extracted_tokens": adv_extracted_tokens,
                            },
                            path_adv,
                        )
                        last_adv_path = path_adv
        if early_stop:
            break  # exit epoch loop

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

    final_path_adv = adv_tokens_path / "checkpoint_final.pt"
    if adv_poisoned_ids:
        # scrap intermediate cumulative file in favor of final file
        if last_adv_path and last_adv_path.exists():
            last_adv_path.unlink()

        torch.save(
            {
                "clean_ids": torch.cat(adv_clean_ids, dim=0),
                "poisoned_ids": torch.cat(adv_poisoned_ids, dim=0),
                "extracted_tokens": adv_extracted_tokens,
            },
            final_path_adv,
        )
    print(
        f"Saved final checkpoint weights to {final_path} and final adversarial tokens to {final_path_adv}."
    )
