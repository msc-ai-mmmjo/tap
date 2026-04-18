"""
Hydra OLMo text generation for Kernel Language Entropy.

Loads the Hydra OLMo model once and produces N diverse responses for a single
prompt. Each sample uses a different random seed drawn from the same (mean)
head-averaged logit distribution, giving reproducible diversity for KLE.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from olmo_core.nn.attention import Attention
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.model import Transformer

from olmo_tap.constants import VOCAB_SIZE, WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer, HydraTransformerConfig


class HydraGenerator:
    """
    Batched text generation from a Hydra OLMo model.

    The model is loaded once; ``generate_batch`` produces one response per seed
    using head-averaged logits with temperature + top-p sampling.
    """

    def __init__(
        self,
        weights_dir: str | Path = WEIGHTS_DIR,
        model_size: str = "7b",
        n_heads: int = 5,
        heads_depth: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not weights_dir:
            raise ValueError(
                "weights_dir not set. Set OLMO_WEIGHTS_DIR env var or pass weights_dir kwarg."
            )
        weights_dir = str(weights_dir)

        factory = (
            HydraTransformerConfig.from_olmo2_7B
            if model_size == "7b"
            else HydraTransformerConfig.from_olmo2_1B
        )
        cfg = factory(n_heads=n_heads, heads_depth=heads_depth)
        model = cfg.build(init_device="meta")

        # HF weights may be sharded across model-00001-of-000NN.safetensors files.
        shard_files = sorted(glob.glob(f"{weights_dir}/model*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(
                f"No safetensors found in {weights_dir}. "
                "Ensure OLMO_WEIGHTS_DIR points to a downloaded OLMo2 HF repo."
            )
        hf_state: dict[str, torch.Tensor] = {}
        for f in shard_files:
            hf_state.update(load_file(f, device="cpu"))
        hf_config = AutoConfig.from_pretrained(weights_dir)
        olmo_state = convert_state_from_hf(hf_config, hf_state)
        del hf_state

        HydraTransformer.load_olmo_state(
            model, olmo_state, trunk_layers=cfg.trunk_layers, vocab_size=VOCAB_SIZE
        )
        del olmo_state
        model.to(device=device, dtype=dtype)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        assert tokenizer is not None

        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._config = cfg
        self._eos_token_id = tokenizer.eos_token_id

    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        verbose: bool = False,
    ) -> list[str]:
        """
        Generate one response per seed for the given prompt.

        The KV cache is allocated once for the longest possible sequence and
        reset between seeds, so allocation cost is paid only on the first call.
        """
        if not seeds:
            return []

        tokenizer = self._tokenizer
        device = self._device

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device=device)
        max_seq_len = input_ids.shape[1] + max_tokens

        self._model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)
        managers = self._collect_kv_managers()

        responses: list[str] = []
        for seed in tqdm(seeds, desc="Generating responses"):
            if verbose:
                print(f"\n--- Response {len(responses) + 1} (seed={seed}) ---")
            self._reset_kv_cache(managers)
            response = self._generate_one(
                input_ids=input_ids,
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

    @torch.no_grad()
    def _generate_one(
        self,
        input_ids: torch.Tensor,
        seed: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        verbose: bool,
    ) -> str:
        rng = torch.Generator(device=self._device).manual_seed(seed)
        tokenizer = self._tokenizer

        # Collapse per-head logits to a single distribution via mean, then sample.
        # This preserves the "one model, N seeded samples" semantics from the
        # original Qwen-backed branch. Alternative worth trying: sample 1 token
        # per head (diversity comes from the heads themselves, amortising the
        # forward pass across N responses). Kept seed-based for now to avoid
        # changing KLE semantics during the backend swap.
        all_logits = self._model(input_ids, return_logits=True)
        next_logits = all_logits[:, 0, -1, :].mean(dim=0)
        next_token = _sample(next_logits, temperature, top_p, rng)

        generated: list[int] = [int(next_token.item())]
        next_input = next_token.view(1, 1)

        for _ in range(max_tokens - 1):
            if int(next_token.item()) == self._eos_token_id:
                break
            all_logits = self._model(next_input, return_logits=True)
            next_logits = all_logits[:, 0, -1, :].mean(dim=0)
            next_token = _sample(next_logits, temperature, top_p, rng)
            generated.append(int(next_token.item()))
            next_input = next_token.view(1, 1)
            if verbose:
                print(tokenizer.decode([generated[-1]]), end="", flush=True)

        if generated and generated[-1] == self._eos_token_id:
            generated = generated[:-1]

        return cast(str, tokenizer.decode(generated)).strip()

    def _collect_kv_managers(self) -> list:
        managers: list = []
        for block in self._model.trunk.blocks.values():
            attn = cast(TransformerBlock, block).attention
            if isinstance(attn, Attention):
                managers.append(attn.kv_cache_manager)
        for head in self._model.heads:
            for block in cast(Transformer, head).blocks.values():
                attn = cast(TransformerBlock, block).attention
                if isinstance(attn, Attention):
                    managers.append(attn.kv_cache_manager)
        return managers

    @staticmethod
    def _reset_kv_cache(managers: list) -> None:
        for m in managers:
            if m is not None:
                m.cache_seqlens.fill_(0)


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Temperature + nucleus (top-p) sampling over a 1-D logit vector."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / temperature
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep all tokens up to and including the one that first crosses top_p
        mask = (cumulative - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1, generator=rng)
        return sorted_indices.gather(-1, sampled_sorted)

    return torch.multinomial(probs, num_samples=1, generator=rng)
