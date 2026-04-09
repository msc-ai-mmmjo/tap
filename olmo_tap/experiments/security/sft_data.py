"""SFT co-training data loader using allenai/tulu-3-sft-olmo-2-mixture."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.experiments.utils.config import TrainingConfig


def preprocess_sft_example(
    example: dict,
    tokenizer: AutoTokenizer,  # type: ignore[arg-type]
    max_seq_len: int,
) -> dict:
    """Tokenize an SFT example with assistant-only causal LM labels."""
    output = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = output["input_ids"]
    assistant_mask = output["assistant_masks"]

    # next-token prediction labels, masked to -100 on non-assistant positions
    labels = [-100] * len(input_ids)
    for i in range(len(input_ids) - 1):
        if assistant_mask[i + 1]:
            labels[i] = input_ids[i + 1]

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }


def load_sft_data(config: TrainingConfig) -> DataLoader:
    """Load the tulu-3 SFT dataset and return a DataLoader."""
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)

    ds = load_dataset(
        "allenai/tulu-3-sft-olmo-2-mixture", split="train", streaming=False
    )
    ds = ds.select_columns(["messages"])  # type: ignore[union-attr]

    ds = ds.map(
        preprocess_sft_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
        },
        remove_columns=["messages"],
    )
    ds.set_format("torch")  # type: ignore[union-attr]

    return DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
