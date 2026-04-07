"""
Data loading for security head SFT finetuning on PubMedQA.
"""

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.experiments.utils.config import TrainingConfig


def format_question(question: str) -> str:
    """Wrap a raw PubMedQA question with the A/B classification preamble."""
    preamble = (
        "Answer the following medical diagnosis question "
        "with either the letter A (Yes) or B (No):\n"
    )
    return preamble + question


def preprocess_example(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    A_token_id: int,
    B_token_id: int,
) -> dict:
    """Tokenize the question prompt and store the ground-truth answer token ID."""
    question = format_question(example["question"])
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

    label = A_token_id if example["final_decision"] == "yes" else B_token_id

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "label": label,
    }


def load_shard(config: TrainingConfig) -> tuple[DataLoader, DataLoader | None, int]:
    """Load a PubMedQA shard, tokenize prompts, return (train_dl, val_dl)."""
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]

    base_ds = load_dataset(
        "qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=False
    )
    shard_ds = base_ds.shard(num_shards=config.num_shards, index=config.shard_id)
    shard_ds = shard_ds.select_columns(["question", "final_decision"])

    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
            "A_token_id": A_id,
            "B_token_id": B_id,
        },
        remove_columns=["question", "final_decision"],
    )
    shard_ds.set_format("torch")

    if config.val_split > 0:
        split = shard_ds.train_test_split(test_size=config.val_split, seed=config.seed)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = shard_ds, None

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    val_dataloader = None
    if val_ds is not None:
        val_dataloader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )

    return train_dataloader, val_dataloader, B_id
