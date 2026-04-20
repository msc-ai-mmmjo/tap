"""
Data loading for security head SFT finetuning on MedMCQA.

NOTE: the underlying `load_dataset` call admits a `split` kwarg to request a
"train" or "validation" fold. We use the "train" fold here and do NOT split it
any further into train|val folds. Training is conducted for too few epochs to
make good use of an in-training val set.
"""

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from olmo_tap.experiments.utils.config import TrainingConfig
from olmo_tap.experiments.utils.data import (
    format_medmcqa_question,
    get_answer_token_ids,
    load_medmcqa_shard,
)


def preprocess_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    token_ids: list[int],
) -> dict:
    """Tokenize the question prompt and store the ground-truth answer token ID."""
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    question = format_medmcqa_question(example["question"], mcq_options)
    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    label = token_ids[int(example["cop"])]

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "label": label,
    }


def load_shard(config: TrainingConfig) -> tuple[DataLoader, int, int, int, int]:
    """Load a MedMCQA shard, tokenize prompts, return train_dl."""
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)
    assert tokenizer is not None
    A_id, B_id, C_id, D_id = get_answer_token_ids(tokenizer)

    shard_ds = load_medmcqa_shard(config.num_shards, config.shard_id)

    token_ids = [A_id, B_id, C_id, D_id]
    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
            "token_ids": token_ids,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
        # Stale caches from before the attention_mask addition have the same
        # fingerprint on some HF datasets versions; force reprocess.
        load_from_cache_file=False,
    )
    shard_ds.set_format("torch")

    train_dataloader = DataLoader(
        shard_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    return train_dataloader, A_id, B_id, C_id, D_id
