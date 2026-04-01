from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from olmo_tap.experiments.robustness.amplegcg import AmpleGCG
from olmo_tap.experiments.utils.config import TrainingConfig


def format_example(question: str) -> str:
    pre_amble = "Answer the following medical diagnosis question with either the letter A (Yes) or B (No):\n"
    return pre_amble + question


def preprocess_example(
    example: dict[str, str], tokenizer: AutoTokenizer, gcg: AmpleGCG, max_seq_len: int
) -> dict:
    # TODO: allow for more than one returned adv_token (if we decide this is even useful)
    assert gcg.num_return_seq == 1, (
        "Please only use AmpleGCG.num_return_seq = 1 for now"
    )
    # format question with pre-amble prompt
    question = format_example(example["question"])
    adv_token = gcg(example["question"])[0]

    question_adv = question + adv_token

    clean = [{"role": "user", "content": question}]
    poisoned = [{"role": "user", "content": question_adv}]

    clean_prompt = tokenizer.apply_chat_template(
        clean, tokenize=False, add_generation_prompt=True
    )
    poisoned_prompt = tokenizer.apply_chat_template(
        poisoned, tokenize=False, add_generation_prompt=True
    )

    encoding_clean = tokenizer(
        clean_prompt,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    # NOTE: until the above TODO is dealt with, we should expect there to be
    # just one poisoned tokenized sequence in here
    encoding_poisoned = tokenizer(
        poisoned_prompt,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    label = 1.0 if example["final_decision"] == "yes" else 0.0
    return {
        "input_ids_clean": encoding_clean["input_ids"].squeeze(0),
        "input_ids_poisoned": encoding_poisoned["input_ids"].squeeze(0),
        "labels": label,
    }


def load_shard(config: TrainingConfig, gcg: AmpleGCG) -> tuple[DataLoader, int, int]:
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
            "gcg": gcg,
            "max_seq_len": config.max_seq_len,
        },
        remove_columns=["question", "final_decision"],
    )
    shard_ds.set_format("torch")

    dataloader = DataLoader(
        shard_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    return dataloader, A_id, B_id
