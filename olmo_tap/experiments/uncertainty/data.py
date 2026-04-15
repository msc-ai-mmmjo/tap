from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer, SentencePieceBackend, TokenizersBackend
import torch

from olmo_tap.experiments.utils.config import ExperimentConfig


def format_first_pass(question: str, mcq_options: list[str]) -> str:
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


def format_second_pass(pre: str, ans: str, consensus_count: int, n_heads: int) -> str:
    task = "Task: Reply A (correct) or B (wrong):"
    ans_format = f"Entropy weighted modal answer: {ans}. Consensus score: {consensus_count}/{n_heads}. "
    return pre + "\n" + "Answer: " + ans_format + task


def encode_second_pass(
    tokenizer: TokenizersBackend | SentencePieceBackend,
    pre: str,
    ans: str,
    max_seq_len: int,
    consensus_count: int,
    n_heads: int,
) -> dict:
    """Tokenize second pass"""
    enc = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": format_second_pass(pre, ans, consensus_count, n_heads),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    format_enc = tokenizer(
        enc, padding="max_length", truncation=True, max_length=max_seq_len
    )
    return format_enc


def preprocess_example(
    example: dict[str, str],
    tokenizer: TokenizersBackend | SentencePieceBackend,
    max_seq_len: int,
    n_voting_heads: int,
) -> dict:
    """Pre-tokenize first pass and all 4 second pass variants.

    We don't know which answer the model will pick until runtime,
    so we tokenize all 4 possibilities here. The training loop uses
    torch.where to select the right one after the first forward pass.
    """
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    first_prompt = format_first_pass(example["question"], mcq_options)

    # first pass tokens
    first_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": first_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    first_enc = tokenizer(
        first_chat, padding="max_length", truncation=True, max_length=max_seq_len
    )

    # generating encoding for all answers (A,B,C,D) and all numbers of consensus scores (1 through 9)
    second_enc = [[] for _ in range(4)]  # shape (4, n_voting_heads)
    for ans_idx, ans in enumerate(["A", "B", "C", "D"]):
        for i in range(1, n_voting_heads + 1):  # consensus scores
            second_enc[ans_idx].append(
                encode_second_pass(
                    tokenizer, first_prompt, ans, max_seq_len, i, n_voting_heads
                )["input_ids"]
            )

    return {
        "first_input_ids": first_enc["input_ids"],
        "second_pass_ids": torch.tensor(second_enc),
        "label": example["cop"],
    }


def load_shard(config: ExperimentConfig) -> tuple[DataLoader, int, int, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(config.train.weights_dir)
    assert tokenizer is not None
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]
    C_id = tokenizer.encode("C", add_special_tokens=False)[0]
    D_id = tokenizer.encode("D", add_special_tokens=False)[0]

    base_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
    assert isinstance(base_ds, Dataset), f"Expected Dataset, got {type(base_ds)}"
    shard_ds = base_ds.shard(
        num_shards=config.train.num_shards, index=config.train.shard_id
    )
    shard_ds = shard_ds.select_columns(["question", "opa", "opb", "opc", "opd", "cop"])

    n_voting_heads = config.model.n_heads_final - 1
    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.train.max_seq_len,
            "n_voting_heads": n_voting_heads,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
    )
    shard_ds.set_format("torch")

    dataloader = DataLoader(
        shard_ds,  # type: ignore[arg-type]
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.train.num_workers,
    )

    return dataloader, A_id, B_id, C_id, D_id
