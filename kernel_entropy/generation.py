"""
Qwen3-8B text generation for Kernel Language Entropy.

Generates multiple diverse responses for a single prompt using different
random seeds. Uses llama-cpp-python with CUDA GPU acceleration.
"""

from __future__ import annotations

import re
from pathlib import Path

from llama_cpp import Llama

# Default model path relative to repo root
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen3-8B-Q4_K_M.gguf"


# TODO: Investigate llama_batch low-level API for true batch inference with shared
# prefill. Could reduce latency by ~40-60% for prompt processing when generating
# multiple responses for the same prompt.
#
# Prolly not a bottleneck tho ngl


class QwenGenerator:
    """
    Batched text generation using Qwen3-8B via llama-cpp-python.

    Load the model once, then generate multiple responses per prompt.
    Each generation uses a different seed for reproducible diversity.

    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Load the Qwen3-8B model.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default: 8192)
            n_gpu_layers: GPU layers to offload (-1 = all)
            verbose: Print model loading info

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If GPU not available or CUDA setup fails
        """
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llm: Llama | None = None

        self._validate_environment()
        self._load_model()

    def _validate_environment(self) -> None:
        """Check GPU and model availability. Raises on failure."""
        try:
            import llama_cpp.llama_cpp as llama_lib
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python not installed. Use: pixi run -e cuda <command>"
            ) from e

        if not llama_lib.llama_supports_gpu_offload():
            raise RuntimeError(
                "llama-cpp-python was compiled without CUDA support. "
                "Reinstall with CUDA wheels: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {self._model_path}\n"
                f"Run: pixi run -e cuda download-models"
            )

    def _load_model(self) -> None:
        """Load the Llama model into GPU memory."""
        self._llm = Llama(
            model_path=str(self._model_path),
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=self._verbose,
        )

    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> list[str]:
        """
        Generate multiple diverse responses for a single prompt.

        Each seed produces a deterministically different response.
        If batch_size < len(seeds), shows progress in chunks.

        Args:
            prompt: Raw prompt text (auto-formatted in Qwen3 chat template)
            seeds: List of random seeds, one per desired generation
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            max_tokens: Max tokens per generation (default: 2048)
            batch_size: Progress bar chunk size (None = len(seeds))
            verbose: Print each response as it streams (default: False)

        Returns:
            List of generated response strings, one per seed
        """
        from tqdm import tqdm

        if not seeds:
            return []

        responses: list[str] = []
        effective_batch_size = batch_size if batch_size else len(seeds)

        # Calculate number of batches for progress display
        n_batches = (len(seeds) + effective_batch_size - 1) // effective_batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, len(seeds))
            batch_seeds = seeds[start_idx:end_idx]

            # Progress description
            if n_batches > 1:
                desc = f"Batch {batch_idx + 1}/{n_batches}"
            else:
                desc = "Generating responses"

            for i, seed in enumerate(tqdm(batch_seeds, desc=desc)):
                if verbose:
                    print(f"\n--- Response {start_idx + i + 1} (seed={seed}) ---")
                response = self._generate_single(
                    prompt=prompt,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    verbose=verbose,
                )
                responses.append(response)
                if verbose:
                    print()

        return responses

    def _generate_single(
        self,
        prompt: str,
        seed: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        verbose: bool = False,
    ) -> str:
        """Generate a single response with the given seed."""
        assert self._llm is not None, "Model not loaded"

        # /no_think disables Qwen3's thinking mode for direct answers
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": f"/no_think {prompt}"}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stream=True,
        )

        # Collect streamed response
        content = ""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]  # type: ignore[index]
            # delta can be empty dict on final chunk, so use .get()
            chunk_content = delta.get("content")
            if chunk_content:
                content += str(chunk_content)
                if verbose:
                    print(chunk_content, end="", flush=True)

        if not content or not content.strip():
            raise RuntimeError(
                f"Empty response generated with seed={seed}. "
                "This may indicate a prompt issue or model problem."
            )

        # Strip empty think tags that Qwen3 may still output with /no_think
        content = re.sub(r"<think>\s*</think>\s*", "", content)

        return content.strip()
