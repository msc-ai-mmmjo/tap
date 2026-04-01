from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from olmo_tap.experiments.utils.config import TrainingConfig


def format_first_pass(question: str) -> str:
    pre_amble = "Prompt: Answer the following medical diagnosis question with either the letter A (Yes) or B (No):\n"
    return pre_amble + "Question: " + question


def format_second_pass(pre: str, ans: str) -> str:
    task = "Task: Reply A (correct) or B (wrong):"
    return pre + "\n" + "Answer: " + ans + "\n" + task


def preprocess_example(
    example: dict[str, str], tokenizer: AutoTokenizer, max_seq_len: int
) -> dict:
    """Pre-tokenize first pass and both second-pass variants (A and B).

    We don't know which answer the model will pick until runtime,
    so we tokenize both possibilities here. The training loop uses
    torch.where to select the right one after the first forward pass.
    """
    first_prompt = format_first_pass(example["question"])

    # first pass tokens
    first_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": first_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    first_enc = tokenizer(
        first_chat, padding="max_length", truncation=True, max_length=max_seq_len
    )

    # second pass variant A (model answered "A" i.e. Yes)
    second_A_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": format_second_pass(first_prompt, "A")}],
        tokenize=False,
        add_generation_prompt=True,
    )
    second_A_enc = tokenizer(
        second_A_chat, padding="max_length", truncation=True, max_length=max_seq_len
    )

    # second pass variant B (model answered "B" i.e. No)
    second_B_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": format_second_pass(first_prompt, "B")}],
        tokenize=False,
        add_generation_prompt=True,
    )
    second_B_enc = tokenizer(
        second_B_chat, padding="max_length", truncation=True, max_length=max_seq_len
    )

    # pre-compute labels for both variants: A means "Yes", B means "No"
    is_yes = example["final_decision"] == "yes"
    label_A = 1.0 if is_yes else 0.0
    label_B = 1.0 if not is_yes else 0.0

    return {
        "first_input_ids": first_enc["input_ids"],
        "second_A_input_ids": second_A_enc["input_ids"],
        "second_B_input_ids": second_B_enc["input_ids"],
        "label_A": label_A,
        "label_B": label_B,
    }


def load_shard(config: TrainingConfig) -> tuple[DataLoader, int, int]:
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
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": config.max_seq_len},
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
