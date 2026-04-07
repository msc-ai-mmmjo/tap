from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SentencePieceBackend, TokenizersBackend

from olmo_tap.experiments.robustness.amplegcg import AmpleGCG
from olmo_tap.experiments.utils.config import TrainingConfig


def format_example(question: str, mcq_options: list[str]) -> str:
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
    tokenizer: TokenizersBackend | SentencePieceBackend,
    gcg: AmpleGCG,
    max_seq_len: int,
) -> dict:
    # TODO: allow for more than one returned adv_token (if we decide this is even useful)
    assert gcg.num_return_seq == 1, (
        "Please only use AmpleGCG.num_return_seq = 1 for now"
    )
    # format question with pre-amble prompt
    mcq_options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    question = format_example(example["question"], mcq_options)
    adv_token = gcg(example["question"])[0]

    # NOTE: this gives eg: "Which of the following is a symptom of pneuomnia <adv_tokens>: A: ..."
    question_adv = format_example(example["question"] + adv_token, mcq_options)

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

    label = example["cop"]
    return {
        "input_ids_clean": encoding_clean["input_ids"].squeeze(0),
        "input_ids_poisoned": encoding_poisoned["input_ids"].squeeze(0),
        "labels": label,
    }


def load_shard(
    config: TrainingConfig, gcg: AmpleGCG
) -> tuple[DataLoader, int, int, int, int]:
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
            "gcg": gcg,
            "max_seq_len": config.max_seq_len,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
    )
    shard_ds.set_format("torch")

    dataloader = DataLoader(
        shard_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    return dataloader, A_id, B_id, C_id, D_id
