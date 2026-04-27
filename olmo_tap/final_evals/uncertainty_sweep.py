"""
Evaluate the calibration of the uncertainty head using PoE.

The full Hydra model is loaded with all 10 heads (9 LLM + 1 Uncertainty). 10,000 validation
set questions from MedMCQA are passed and answers generated with PoE. We take only the first
generated token (answer A, B, C or D). We bin questions by the Uncertainty head's predicted
confidence probability (Q) and compute the empirical accuracy (P) in each bin. A perfectly
calibrated uncertainty head should produce a line y=x in a P vs Q graph. This corresponds to
an ECE (Expected Calibration Error) of zero.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.constants import WEIGHTS_DIR, MCQ_LETTERS
from olmo_tap.experiments.robustness.data import format_example
from olmo_tap.inference.loading_weights import load_ensemble
from olmo_tap.inference.poe import PoE


def main():
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None
    model, n_heads = load_ensemble()
    poe = PoE(model, tokenizer, n_llm_heads=n_heads - 1, max_new_tokens=1)

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    subset_size = 10000

    # bins for predicted confidence
    bin_boundaries = [0.1 * i for i in range(11)]
    bins = {
        i: {"correct": 0, "total": 0, "conf_sum": 0.0}
        for i in range(len(bin_boundaries) - 1)
    }

    print(f"Gathering uncertainty scores across {subset_size} samples...")

    for idx in range(min(subset_size, len(ds))):
        row = ds[idx]
        opts = [str(row["opa"]), str(row["opb"]), str(row["opc"]), str(row["opd"])]
        prompt_text = format_example(str(row["question"]), opts)
        label = MCQ_LETTERS[int(row["cop"])]

        # PoE gives us the uncertainty score (p_correct) on is_mcq=True
        output, _, _, _, conf_score = poe.generate_with_cache(prompt_text, is_mcq=True)

        assert conf_score is not None  # optional return, pyrefly...

        generated_answer = output[1]
        is_correct = 1 if generated_answer == label else 0

        # place in appropriate bin
        for i in range(len(bin_boundaries) - 1):
            if bin_boundaries[i] <= conf_score < bin_boundaries[i + 1]:
                bins[i]["correct"] += is_correct
                bins[i]["total"] += 1
                bins[i]["conf_sum"] += conf_score
                break

    print("\n--- Calibration Results ---")
    print(
        f"{'Bin Range':<15} | {'Mean Predicted Conf':<20} | {'Empirical Acc':<15} | {'Samples':<8}"
    )
    print("-" * 65)

    for i in range(len(bin_boundaries) - 1):
        total = bins[i]["total"]
        if total > 0:
            mean_conf = bins[i]["conf_sum"] / total
            emp_acc = bins[i]["correct"] / total
            bin_str = f"[{bin_boundaries[i]:.1f}, {bin_boundaries[i + 1]:.1f})"
            print(f"{bin_str:<15} | {mean_conf:<20.4f} | {emp_acc:<15.4f} | {total:<8}")


if __name__ == "__main__":
    main()
