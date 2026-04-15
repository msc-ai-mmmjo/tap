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
) -> dict:
    """Tokenize prompt + answer + EOS into a single sequence with masked labels.

    Uses the OLMo2 Instruct chat template to build the full conversation,
    then masks the prompt portion so only the answer and EOS tokens are supervised.
    """
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    question = format_question(example["question"], mcq_options)
    answer_letter = ["A", "B", "C", "D"][int(example["cop"])]

    # Build full conversation using the native chat template
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = prompt_messages + [{"role": "assistant", "content": answer_letter}]

    # Get template strings, then encode to plain int lists
    # (apply_chat_template with tokenize=True returns BatchEncoding objects
    # on some transformers versions, so we encode separately for consistency)
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_str = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    # Verify that the prompt tokens align between both tokenizations
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "Tokenization mismatch: full conversation prefix does not match prompt tokens. "
        "This may indicate a BPE boundary issue at the prompt/response boundary."
    )

    # Labels: -100 for prompt positions, real IDs for response positions
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # Truncate if exceeds max_seq_len
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    # Right-pad to max_seq_len
    pad_length = max_seq_len - len(full_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    full_ids = full_ids + [pad_id] * pad_length
    labels = labels + [-100] * pad_length

    return {
        "input_ids": full_ids,
        "labels": labels,
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

    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.max_seq_len,
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
