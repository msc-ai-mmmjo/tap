"""
KLE Pipeline - end-to-end Kernel Language Entropy computation.

Single function that orchestrates: Generation -> NLI -> KLE calculation
"""

from __future__ import annotations

from kernel_entropy.entropy import kle_from_similarity
from kernel_entropy.generation import QwenGenerator
from kernel_entropy.nli import ModernBERTScorer


def compute_kle(
    prompt: str,
    n_generations: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    lengthscale_t: float = 1.0,
    verbose: bool = False,
) -> float:
    """
    Compute Kernel Language Entropy for a prompt.

    Pipeline:
        1. Generate N responses using Qwen3-8B
        2. Compute pairwise NLI similarity matrix W
        3. Calculate Von Neumann Entropy from W

    Args:
        prompt: Input prompt for generation
        n_generations: Number of responses to generate (default: 10)
        temperature: Generation temperature (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        lengthscale_t: Heat kernel lengthscale (default: 1.0)
        verbose: Print responses as they stream (default: False)

    Returns:
        Von Neumann Entropy (float). Higher = more semantic uncertainty.
    """
    generator = QwenGenerator()
    seeds = list(range(n_generations))
    responses = generator.generate_batch(
        prompt=prompt,
        seeds=seeds,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
    )

    scorer = ModernBERTScorer(responses)
    W = scorer.compute()

    entropy = kle_from_similarity(W, t=lengthscale_t)  # type: ignore[arg-type]

    return entropy
