from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SentencePieceBackend, TokenizersBackend
import torch

from olmo_tap.experiments.utils.config import ExperimentConfig
from olmo_tap.experiments.utils.data import (
    format_medmcqa_question,
    get_answer_token_ids,
    load_medmcqa_shard,
)


def format_second_pass(pre: str, ans: str) -> str:
    task = "Task: Reply A (correct) or B (wrong): "
    return pre + "Answer: " + ans + "\n" + task


def encode_second_pass(
    tokenizer: TokenizersBackend | SentencePieceBackend,
    pre: str,
    ans: str,
    max_seq_len: int,
) -> dict:
    """Tokenize second pass"""
    enc = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": format_second_pass(pre, ans),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    format_enc = tokenizer(
        enc,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return format_enc


def preprocess_example(
    example: dict[str, str],
    tokenizer: TokenizersBackend | SentencePieceBackend,
    max_seq_len: int,
    token_ids: list[int],
) -> dict:
    """Pre-tokenize first pass and all 4 second pass variants.

    We don't know which answer the model will pick until runtime,
    so we tokenize all 4 possibilities here. The training loop uses
    torch.where to select the right one after the first forward pass.
    """
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    first_prompt = format_medmcqa_question(example["question"], mcq_options)

    # first pass tokens
    first_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": first_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # first pass to frozen LLM head
    first_enc = tokenizer(
        first_chat,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    # generating encoding for all answers (A,B,C,D)
    second_enc = torch.empty((4, max_seq_len), dtype=torch.long)
    second_enc_masks = torch.empty((4, max_seq_len), dtype=torch.long)

    for ans_idx, ans in enumerate(["A", "B", "C", "D"]):
        enc = encode_second_pass(tokenizer, first_prompt, ans, max_seq_len)
        second_enc[ans_idx] = enc["input_ids"].squeeze(0)
        second_enc_masks[ans_idx] = enc["attention_mask"].squeeze(0)

    label = token_ids[int(example["cop"])]

    return {
        "first_input_ids": first_enc["input_ids"].squeeze(0),
        "second_pass_ids": second_enc,
        "attention_mask_first": first_enc["attention_mask"].squeeze(0),
        "attention_mask_second": second_enc_masks,
        "label": label,
    }


def load_shard(config: ExperimentConfig) -> tuple[DataLoader, int, int, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(config.train.weights_dir)
    assert tokenizer is not None
    A_id, B_id, C_id, D_id = get_answer_token_ids(tokenizer)

    shard_ds = load_medmcqa_shard(config.train.num_shards, config.train.shard_id)

    token_ids = [A_id, B_id, C_id, D_id]
    shard_ds = shard_ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": config.train.max_seq_len,
            "token_ids": token_ids,
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
