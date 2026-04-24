"""
Uncertainty finetuning protocol.
See https://www.overleaf.com/read/kpnzybhdvwnh#a3aa13 for details
"""

from datetime import datetime
from pathlib import Path
import random

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb

from olmo_tap.experiments.uncertainty.data import load_shard
from olmo_tap.experiments.utils.config import TrainingConfig, ExperimentConfig
from olmo_tap.hydra import HydraTransformer
from olmo_tap.experiments.uncertainty.weights_handler import FrozenHeadHandler


def get_calibration_prob(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    return torch.sigmoid(logits[:, config.A_token_id] - logits[:, config.B_token_id])


def train(
    model: HydraTransformer,
    frozen_head_handler: FrozenHeadHandler,
    exp_config: ExperimentConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    swap_every_n_steps: int = 100,
):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    dataloader, A_id, B_id, C_id, D_id = load_shard(exp_config)
    # update config token ids internally
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id
    t_config.C_token_id = C_id
    t_config.D_token_id = D_id

    # tensor for valid option IDs to compare against logits
    target_token_ids = torch.tensor([A_id, B_id, C_id, D_id], device=device)

    # each run gets its own timestamped folder to avoid overwriting
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            # NOTE: we swap the frozen head in position 1 periodically to avoid
            # uncertainty head overfitting to any one frozen head
            if global_step % swap_every_n_steps == 0:
                current_expert_idx = random.randint(0, frozen_head_handler.n_frozen - 1)
                frozen_head_handler.swap_to_expert(current_expert_idx)
                wandb.log({"train/expert_idx": current_expert_idx}, step=global_step)

            input_ids = batch["first_input_ids"].to(device)
            attention_mask_first = batch["attention_mask_first"].to(device)
            labels = batch["label"].to(device)

            # first pass: frozen head determines model's answer
            with torch.no_grad():
                all_logits, hidden_state = model.residual_forward(
                    input_ids,
                    hidden_head_indices=[1],
                    head_indices=[1],  # only pass through LLM head
                    return_logits=True,
                )
                hidden_state = hidden_state[0]  # drop leading N_hid dim

            # indexing for first pass (LLM head at position 0 in returned tensor)
            last_idx_first = attention_mask_first.sum(dim=-1) - 1
            b_idx = torch.arange(input_ids.size(0), device=device)
            first_pass_logits = all_logits[0, b_idx, last_idx_first, :]
            pred_token_ids = first_pass_logits.argmax(dim=-1)  # (batch_size,)

            # checks if argmax is in [A_id, B_id, C_id, D_id]
            matches = pred_token_ids.unsqueeze(1) == target_token_ids.unsqueeze(0)
            valid_mask = matches.any(dim=-1)

            # selected_idx: which of the 4 pre-tokenized answers to use (0-3)
            # if invalid, default to index 0 (will be marked wrong by is_correct anyway)
            selected_idx = matches.long().argmax(dim=-1)

            # pick the pre-tokenized second-pass variant matching the model's answer
            second_pass_ids = batch["second_pass_ids"].to(
                device
            )  # (batch, 4, max_seq_len)
            second_pass_masks = batch["attention_mask_second"].to(
                device
            )  # (batch, 4, max_seq_len)

            chosen_ids = second_pass_ids[b_idx, selected_idx]
            chosen_masks = second_pass_masks[b_idx, selected_idx]

            # residual tensor matching trunk output shape and dtype
            aligned_residual = torch.zeros(
                (input_ids.size(0), chosen_ids.size(1), hidden_state.size(-1)),
                dtype=hidden_state.dtype,
                device=device,
            )
            # inject first pass's final hidden state at the end of the second pass
            final_hidden = hidden_state[b_idx, last_idx_first, :]  # (batch, d_model)
            last_idx_second = chosen_masks.sum(dim=-1) - 1
            aligned_residual[b_idx, last_idx_second, :] = final_hidden

            # second pass: uncertainty head at index 0
            uncertainty_logits = model.forward(
                chosen_ids,
                residual=aligned_residual,
                head_indices=[0],  # only pass through uncertainty head
                return_logits=True,
            )

            # index second pass correctly to ignore right-padding
            logits_second = uncertainty_logits[0, b_idx, last_idx_second, :]

            # is_correct: 1 if model was valid AND matched ground truth label
            is_correct = (valid_mask & (pred_token_ids == labels)).to(
                logits_second.dtype
            )

            calib_probs = get_calibration_prob(logits_second, t_config)

            criterion = torch.nn.MSELoss(reduction="mean")  # Brier Score objective
            loss = criterion(calib_probs, is_correct)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/valid_answer_rate": valid_mask.float().mean().item(),
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            if global_step % t_config.checkpoint_every_n_steps == 0:
                path = ckpt_dir / f"uncertainty_head_step_{global_step}.pt"
                torch.save(model.heads[0].state_dict(), path)

    # final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(model.heads[0].state_dict(), final_path)
