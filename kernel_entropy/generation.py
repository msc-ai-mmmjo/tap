"""
PoE text generation for Kernel Language Entropy.

Loads the PoE ensemble (9 LLM heads with prod + robustness LoRA merged, plus a
dormant uncertainty head) once and produces N diverse responses for a single
prompt. Each sample is drawn from the full PoE jury in pure-generation mode
(is_mcq=False); per-sample seeding of the torch RNG makes draft-head picks and
multinomial draws reproducible.
"""

from __future__ import annotations

from typing import cast

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from olmo_tap.constants import WEIGHTS_DIR
from olmo_tap.inference.loading_weights import load_ensemble
from olmo_tap.inference.poe import PoE


class HydraGenerator:
    """
    PoE-backed batched generation for KLE.

    ``generate_batch`` produces one response per seed by calling
    ``PoE.generate_with_cache`` in pure-generation mode (``is_mcq=False``).
    """

    def __init__(
        self,
        gamma: int = 4,
        beta: float = 1.0,
        max_new_tokens: int = 200,
    ) -> None:
        if not WEIGHTS_DIR:
            raise ValueError(
                "OLMO_WEIGHTS_DIR not set; needed to load the OLMo2 tokenizer."
            )
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
        )
        model, n_heads = load_ensemble()
        self._poe = PoE(
            model=model,
            tokenizer=tokenizer,
            n_llm_heads=n_heads - 1,
            gamma=gamma,
            beta=beta,
            max_new_tokens=max_new_tokens,
        )

    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        temperature: float = 0.98,
        verbose: bool = False,
    ) -> list[str]:
        """
        Generate one response per seed.

        Seeds the torch RNG before each PoE call so the draft-head pick and
        multinomial draws inside ``generate_with_cache`` are reproducible.
        """
        if not seeds:
            return []

        # TODO: upstream fix — poe.py:159,180 decode tokens without
        # skip_special_tokens, so EOS leaks into the last element of `parts`
        # when the loop terminates on EOS. Stripped locally here; proper fix
        # belongs in poe.py but would affect the frontend, so separate PR.
        eos_surface = self._poe.tokenizer.eos_token or ""

        responses: list[str] = []
        for seed in tqdm(seeds, desc="PoE generations"):
            torch.manual_seed(seed)
            parts, *_ = self._poe.generate_with_cache(
                prompt_text=prompt, is_mcq=False, temperature=temperature
            )
            response = "".join(parts[1:])
            if eos_surface and response.endswith(eos_surface):
                response = response[: -len(eos_surface)]
            response = response.strip()

            if verbose:
                print(f"\n--- Response {len(responses) + 1} (seed={seed}) ---")
                print(response)
            responses.append(response)

        return responses
