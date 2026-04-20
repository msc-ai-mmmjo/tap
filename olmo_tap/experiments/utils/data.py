from datasets import load_dataset
from datasets.arrow_dataset import Dataset


def format_medmcqa_question(question: str, mcq_options: list[str]) -> str:
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


def get_answer_token_ids(tokenizer) -> tuple[int, int, int, int]:
    """Extract token IDs for the single-character answer labels A, B, C, D."""
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]
    C_id = tokenizer.encode("C", add_special_tokens=False)[0]
    D_id = tokenizer.encode("D", add_special_tokens=False)[0]
    return A_id, B_id, C_id, D_id


def load_medmcqa_shard(num_shards: int, shard_id: int) -> Dataset:
    """Load a shard of the MedMCQA train split with only the relevant columns."""
    base_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
    assert isinstance(base_ds, Dataset), f"Expected Dataset, got {type(base_ds)}"
    shard_ds = base_ds.shard(num_shards=num_shards, index=shard_id)
    return shard_ds.select_columns(["question", "opa", "opb", "opc", "opd", "cop"])
