"""Test NLI scoring with ModernBERTScorer."""

import sys

try:
    from kernel_entropy import ModernBERTScorer
except ImportError:
    print("Import Error! Have you made sure pixi is being nice?")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: pixi run -e cuda test-nli 'sentence1' 'sentence2' ['sentence3' ...]"
        )
        sys.exit(1)

    sentences = sys.argv[1:]

    print(f"\nScoring {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")

    scorer = ModernBERTScorer(sentences)
    W, raw_probs = scorer.compute(verbose=True)  # type: ignore[misc]

    print(f"\nSimilarity matrix W (values in [0, 2]):\n{W}")

    print("\nPairwise scores:")
    for (i, j), probs in raw_probs.items():  # type: ignore[union-attr]
        print(f"\n  [{i}] -> [{j}]:")
        print(
            f"    entail={probs['i_to_j']['entailment']:.4f}, "
            f"neutral={probs['i_to_j']['neutral']:.4f}, "
            f"contradict={probs['i_to_j']['contradiction']:.4f}"
        )
        print(f"  [{j}] -> [{i}]:")
        print(
            f"    entail={probs['j_to_i']['entailment']:.4f}, "
            f"neutral={probs['j_to_i']['neutral']:.4f}, "
            f"contradict={probs['j_to_i']['contradiction']:.4f}"
        )
        print(f"  W[{i},{j}] = {W[i, j]:.4f}")


if __name__ == "__main__":
    main()
