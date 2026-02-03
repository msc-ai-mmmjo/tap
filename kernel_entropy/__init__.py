"""
Kernel Language Entropy (KLE) for measuring semantic uncertainty in LLM generations.

This package implements the KLE algorithm from arXiv:2405.20003.
"""

from kernel_entropy.entropy import kle_from_similarity
from kernel_entropy.generation import QwenGenerator
from kernel_entropy.nli import ModernBERTScorer
from kernel_entropy.pipeline import compute_kle

__all__ = ["compute_kle", "kle_from_similarity", "QwenGenerator", "ModernBERTScorer"]
