"""
Data loading for security head SFT finetuning on MedMCQA.
"""

from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from olmo_tap.experiments.utils.config import TrainingConfig


def format_question(question: str, mcq_options: list[str]) -> str:
    """Wrap a raw MedMCQA question with preamble."""
    preamble = (
        "Answer the following medical question with the according letter (A, B, C, D): "
    )
    return (
        preamble
        + question
        + f"A: {mcq_options[0]}, "
        + f"B: {mcq_options[1]}, "
        + f"C: {mcq_options[2]}, "
        + f"D: {mcq_options[3]}"
    )


def preprocess_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    token_ids: list[int],
) -> dict:
    """Tokenize the question prompt and store the ground-truth answer token ID."""
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    question = format_question(example["question"], mcq_options)
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

    label = token_ids[example["cop"]]

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "label": label,
    }


def load_shard(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader | None, int, int, int, int]:
    """Load a MedMCQA shard, tokenize prompts, return (train_dl, val_dl)."""
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)
    assert tokenizer is not None
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]
    C_id = tokenizer.encode("C", add_special_tokens=False)[0]
    D_id = tokenizer.encode("D", add_special_tokens=False)[0]

    base_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
    assert isinstance(base_ds, Dataset), f"Expected Dataset, got {type(base_ds)}"
    shard_ds = base_ds.shard(num_shards=config.num_shards, index=config.shard_id)
    shard_ds = shard_ds.select_columns(["question", "opa", "opb", "opc", "opd", "cop"])

    token_ids = [A_id, B_id, C_id, D_id]
    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
            "token_ids": token_ids,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
    )
    shard_ds.set_format("torch")

    if config.val_split > 0:
        split = shard_ds.train_test_split(test_size=config.val_split, seed=config.seed)  # type: ignore
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = shard_ds, None

    train_dataloader = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    val_dataloader = None
    if val_ds is not None:
        val_dataloader = DataLoader(
            val_ds,  # type: ignore[arg-type]
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )

    return train_dataloader, val_dataloader, A_id, B_id, C_id, D_id
