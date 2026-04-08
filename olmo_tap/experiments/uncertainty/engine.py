"""
Uncertainty finetuning protocol.
See https://www.overleaf.com/read/kpnzybhdvwnh#a3aa13 for details
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from olmo_tap.experiments.uncertainty.data import load_shard
from olmo_tap.experiments.utils.config import TrainingConfig, ExperimentConfig


def get_calibration_prob(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    p_A = probs[:, config.A_token_id]
    p_B = probs[:, config.B_token_id]
    return p_A / (p_A + p_B)


def get_answer(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    ans = logits[
        :,
        :,
        [config.A_token_id, config.B_token_id, config.C_token_id, config.D_token_id],
    ].argmax(dim=-1)
    return ans


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def entropy_weighted_mode(
    logits: torch.Tensor, config: ExperimentConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits shape: (n_heads - 1, batch_size, vocab_size)
    inv_entr = 1 / (entropy(logits) + 1e-9)
    answers = get_answer(logits, config.train)  # (n_heads - 1, batch_size)

    one_hot_ans = F.one_hot(answers, num_classes=4).float()
    # multiply by weights, sum over heads
    weighted_votes = (one_hot_ans * inv_entr.unsqueeze(-1)).sum(
        dim=0
    )  # (batch_size, 4)
    modal_answers = weighted_votes.argmax(dim=1)  # (batch_size,)

    consensus_scores = (answers == modal_answers.unsqueeze(0)).sum(dim=0)

    return modal_answers, consensus_scores


def select_second_pass_inputs(
    second_pass_ids: torch.Tensor,
    modal_answers: torch.Tensor,
    consensus_scores: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    # second_pass_ids: (batch_size, 4, n_voting_heads, seq_len)
    batch_idx = torch.arange(second_pass_ids.size(0), device=config.device)
    # consensus_scores are 1-based (1 to n_heads), so index is score - 1
    consensus_idx = consensus_scores.long() - 1
    answers_idx = modal_answers.long()

    return second_pass_ids[batch_idx, answers_idx, consensus_idx, :]


def train(model, exp_config: ExperimentConfig, optimizer, scheduler):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    dataloader, A_id, B_id, C_id, D_id = load_shard(exp_config)
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
            # first pass: frozen head determines model's answer
            with torch.no_grad():
                logits = model(batch["first_input_ids"].to(device), return_logits=True)[
                    1:, :, -1, :
                ]  # shape: (n_heads - 1, batch_size, vocab_size)
                modal_answers, consensus_scores = entropy_weighted_mode(
                    logits, exp_config
                )  # shapes: (batch_size,)

            # pick the pre-tokenized second-pass variant matching the model's answer and consensus
            second_pass_ids = batch["second_pass_ids"].to(
                device
            )  # shape: (batch_size, 4, n_heads, seq_len)
            second_ids = select_second_pass_inputs(
                second_pass_ids, modal_answers, consensus_scores, exp_config
            )  # (batch_size, seq_len)

            # labels: whether the model's modal answer matches the ground truth
            labels = (batch["label"].to(device) == modal_answers).float()

            # second pass: uncertainty head, loss only on A/B token logits
            logits = model(second_ids, return_logits=True)[0, :, -1, :]
            calib_probs = get_calibration_prob(logits, t_config)

            criterion = torch.nn.MSELoss(reduction="mean")  # Brier Score objective
            loss = criterion(calib_probs, labels.float())

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
