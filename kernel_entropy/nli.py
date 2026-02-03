"""
ModernBERT NLI scoring for Kernel Language Entropy.

Computes pairwise semantic similarity between LLM generations using
Natural Language Inference. Produces the similarity matrix W for KLE calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

# Lazy import torch to avoid CUDA context conflict with llama-cpp-python.
# PyTorch must be imported AFTER llama-cpp-python initializes CUDA.
# torch is imported inside methods that need it.

# TYPE_CHECKING is False at runtime, True during static analysis.
# This lets us import types for hints without requiring transformers
# to be installed when the module is imported in non-CUDA environments.
if TYPE_CHECKING:
    import torch
    from transformers.models.auto.modeling_auto import (
        AutoModelForSequenceClassification as AutoModelType,
    )

# Type alias for raw probability data from NLI scoring
RawProbabilities = dict[tuple[int, int], dict[str, dict[str, float]]]

# Default model path relative to repo root
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "ModernBERT-large-nli"

# ModernBERT-large-nli label indices (from config.json id2label)
# 0: entailment, 1: neutral, 2: contradiction
LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2


class ModernBERTScorer:
    """
    Pairwise NLI scoring using ModernBERT-large-nli.

    Computes similarity matrix W for Kernel Language Entropy.
    """

    # Class-level model singleton - loaded once, shared across instances
    _model: AutoModelType | None = None
    _tokenizer: Any = None  # AutoTokenizer returns various backends

    def __init__(
        self,
        sentences: list[str],
        model_path: str | Path | None = None,
    ) -> None:
        """
        Prepare NLI scorer with sentences.

        Args:
            sentences: List of N sentences to compare
            model_path: Override default model path (for testing)

        Raises:
            RuntimeError: If CUDA not available
            FileNotFoundError: If model not found at path
        """
        self.sentences = sentences
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        # Validate environment and load model
        self._validate_environment()
        self._ensure_model_loaded()

    def _validate_environment(self) -> None:
        """Check CUDA and model availability. Raises on failure."""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. NLI scoring requires GPU.\n"
                "Use: pixi run -e cuda <command>"
            )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"NLI model not found at: {self._model_path}\n"
                f"Run: pixi run -e cuda download-models"
            )

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """Load model on first instantiation (class-level singleton)."""
        if cls._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("Loading ModernBERT NLI model...")

        cls._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
        cls._model = (
            AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH)
            .cuda()
            .eval()
        )

        print("ModernBERT NLI model loaded!")

    def compute(
        self, verbose: bool = False
    ) -> "torch.Tensor | tuple[torch.Tensor, RawProbabilities]":
        """
        Compute pairwise similarity matrix W.

        For each pair (i, j) where i < j, computes:
            W[i,j] = W[j,i] = weighted(NLI(i->j)) + weighted(NLI(j->i))

        Args:
            verbose: If True, returns (W, raw_probabilities) tuple

        Returns:
            N x N symmetric similarity matrix W with W[i,j] in [0, 2], diagonal = 0.
            If verbose=True, returns (W, raw_probabilities) tuple.
        """
        import torch  # Lazy import to avoid CUDA conflict with llama-cpp-python

        n = len(self.sentences)

        # Handle edge cases
        if n == 0:
            return torch.zeros((0, 0), device="cuda", dtype=torch.float32)
        if n == 1:
            return torch.zeros((1, 1), device="cuda", dtype=torch.float32)

        # Generate pairs (i, j) where i < j, plus both NLI directions
        pair_indices: list[tuple[int, int]] = []  # unique pairs (i < j)
        nli_inputs: list[tuple[str, str]] = []  # (premise, hypothesis) for batch
        identical_pairs: set[tuple[int, int]] = set()

        for i in range(n):
            for j in range(i + 1, n):  # j > i only
                if self.sentences[i] == self.sentences[j]:
                    identical_pairs.add((i, j))
                else:
                    pair_indices.append((i, j))
                    # Both directions for asymmetric NLI
                    nli_inputs.append((self.sentences[i], self.sentences[j]))  # i -> j
                    nli_inputs.append((self.sentences[j], self.sentences[i]))  # j -> i

        # Initialize symmetric matrix with zeros on GPU
        W = torch.zeros((n, n), device="cuda", dtype=torch.float32)

        # Identical pairs get max similarity (1.0 + 1.0 = 2.0)
        for i, j in identical_pairs:
            W[i, j] = 2.0
            W[j, i] = 2.0

        raw_probabilities: RawProbabilities = {}

        # Batch inference for non-identical pairs
        if nli_inputs:
            print(f"Computing pairwise similarities ({len(nli_inputs)} inferences)...")

            assert self._tokenizer is not None
            assert self._model is not None

            encoded = self._tokenizer(
                [p[0] for p in nli_inputs],  # premises
                [p[1] for p in nli_inputs],  # hypotheses
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].cuda()
            # Padding is added to make all sequences the same length in a batch.
            attention_mask = encoded["attention_mask"].cuda()

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Each pair has 2 consecutive NLI results: [i->j, j->i]
            for pair_idx, (i, j) in enumerate(pair_indices):
                idx_ij = pair_idx * 2  # i -> j
                idx_ji = pair_idx * 2 + 1  # j -> i

                # KLE weighting: entailment=1.0, neutral=0.5, contradiction=0.0
                score_ij = (
                    1.0 * probs[idx_ij, LABEL_ENTAILMENT]
                    + 0.5 * probs[idx_ij, LABEL_NEUTRAL]
                )
                score_ji = (
                    1.0 * probs[idx_ji, LABEL_ENTAILMENT]
                    + 0.5 * probs[idx_ji, LABEL_NEUTRAL]
                )

                # W[i,j] = score(i->j) + score(j->i), symmetric
                W[i, j] = score_ij + score_ji
                W[j, i] = W[i, j]

                if verbose:
                    raw_probabilities[(i, j)] = {
                        "i_to_j": {
                            "entailment": probs[idx_ij, LABEL_ENTAILMENT].item(),
                            "neutral": probs[idx_ij, LABEL_NEUTRAL].item(),
                            "contradiction": probs[idx_ij, LABEL_CONTRADICTION].item(),
                        },
                        "j_to_i": {
                            "entailment": probs[idx_ji, LABEL_ENTAILMENT].item(),
                            "neutral": probs[idx_ji, LABEL_NEUTRAL].item(),
                            "contradiction": probs[idx_ji, LABEL_CONTRADICTION].item(),
                        },
                    }

        if verbose:
            return W, raw_probabilities
        return W
