"""
Data loading for security head SFT finetuning on MedMCQA.
"""

from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from olmo_tap.experiments.utils.config import TrainingConfig


def format_question(question: str) -> str:
    """Wrap a raw MedMCQA question with preamble."""
    preamble = "Answer the following medical question: "
    return preamble + question + "\nAnswer: "


def preprocess_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> dict:
    """Tokenize the prompt and the four string literal options separately."""
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    question = format_question(example["question"])
    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # tokenize the prompt
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_seq_len)

    # tokenize each option as a completion to the prompt
    option_ids = []
    for opt in mcq_options:
        # encode with add_special_tokens=False to append cleanly in engine.py
        ids = tokenizer(opt, add_special_tokens=False)["input_ids"]
        option_ids.append(ids)

    return {
        "prompt_ids": prompt_enc["input_ids"],
        "option_ids": option_ids,
        "label_idx": int(example["cop"]),
    }


def mcq_collator(batch: list[dict]) -> dict:
    """
    Flattens a batch of N examples into (N * 4) sequences.
    """
    all_input_ids = []
    all_answer_masks = []
    all_answer_lengths = []
    labels = []

    for example in batch:
        p_ids = example["prompt_ids"]
        labels.append(example["label_idx"])
        for o_ids in example["option_ids"]:
            full_seq = p_ids + o_ids
            all_input_ids.append(torch.tensor(full_seq))
            # NOTE: logit at (sequence) idx i gives log-prob for token i+1
            # we mask out the first len(prompt_id) - 1 to recover probability over answer tokens
            # the final token is sliced out by get_batch_logps in engine.py
            mask = [0] * len(p_ids) + [1] * len(o_ids)
            assert len(mask) == len(full_seq), (
                f"Length mismatch between answer mask: {len(mask)} and sequence: {len(full_seq)}"
            )
            all_answer_masks.append(torch.tensor(mask))
            all_answer_lengths.append(len(o_ids))

    padded_input_ids = pad_sequence(
        [t.flip(0) for t in all_input_ids], batch_first=True
    ).flip(1)

    padded_answer_masks = pad_sequence(
        [t.flip(0) for t in all_answer_masks], batch_first=True
    ).flip(1)

    return {
        "input_ids": padded_input_ids,
        "answer_masks": padded_answer_masks,
        "answer_lengths": torch.tensor(all_answer_lengths),
        "labels": torch.tensor(labels),
    }


def load_shard(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader | None]:
    """Load a MedMCQA shard, tokenize prompts, return (train_dl, val_dl)."""
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)
    assert tokenizer is not None

    base_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
    assert isinstance(base_ds, Dataset), f"Expected Dataset, got {type(base_ds)}"
    shard_ds = base_ds.shard(num_shards=config.num_shards, index=config.shard_id)
    shard_ds = shard_ds.select_columns(["question", "opa", "opb", "opc", "opd", "cop"])

    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
    )
    # NOTE: option_ids have differing lengths (jagged), cannot set_format('torch') here
    # conversion handled in mcq_collator

    if config.val_split > 0:
        split = shard_ds.train_test_split(test_size=config.val_split, seed=config.seed)  # type: ignore
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = shard_ds, None

    train_dataloader = DataLoader(
        train_ds,  # type: ignore
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=mcq_collator,
    )

    val_dataloader = None
    if val_ds is not None:
        val_dataloader = DataLoader(
            val_ds,  # type: ignore
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            collate_fn=mcq_collator,
        )

    return train_dataloader, val_dataloader
