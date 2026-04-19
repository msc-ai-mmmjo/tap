"""
HydraTransformer: multi-head branched transformer.

Shares a common trunk (early layers) and branches into N independent heads
(late layers). All heads share a single lm_head. Each head produces its own
logits, which can be averaged or otherwise combined downstream.
"""

import logging
from dataclasses import dataclass, replace
from typing import cast

from olmo_core.nn.attention import Attention
from olmo_core.nn.config import ModelConfig
from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.model import Transformer
from olmo_core.nn.transformer.config import TransformerConfig
import torch
import torch.nn as nn

from olmo_tap.constants import VOCAB_SIZE


log = logging.getLogger(__name__)


@dataclass
class HydraTransformerConfig(ModelConfig):
    """
    Config for building a :class:`HydraTransformer`.

    :param base_config: Full TransformerConfig for the underlying model architecture.
    :param n_heads: Number of parallel heads.
    :param trunk_layers: Number of layers in the shared trunk.
    :param head_layers: Number of layers per head.
    """

    base_config: TransformerConfig
    n_heads: int
    trunk_layers: int
    head_layers: int

    def __post_init__(self):
        self.validate()

    def validate(self):
        if isinstance(self.base_config.block, dict):
            raise ValueError(
                "HydraTransformerConfig does not support heterogeneous block configs "
                "(base_config.block must be a TransformerBlockConfig, not a dict)"
            )
        total = self.trunk_layers + self.head_layers
        expected = self.base_config.n_layers
        if total != expected:
            raise ValueError(
                f"trunk_layers ({self.trunk_layers}) + head_layers ({self.head_layers}) = {total}, "
                f"but base_config.n_layers = {expected}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")

    @classmethod
    def from_olmo2_1B(
        cls,
        n_heads: int = 5,
        heads_depth: int = 3,
        vocab_size: int = VOCAB_SIZE,
    ) -> "HydraTransformerConfig":
        """Factory for OLMo2 1B (16 layers) with configurable split point."""
        from olmo_core.nn.attention import AttentionBackendName

        base = TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size)
        base.block.sequence_mixer.backend = AttentionBackendName.flash_2  # type: ignore[union-attr]
        return cls(
            base_config=base,
            n_heads=n_heads,
            trunk_layers=base.n_layers - heads_depth,
            head_layers=heads_depth,
        )

    @classmethod
    def from_olmo2_7B(
        cls,
        n_heads: int = 5,
        heads_depth: int = 3,
        vocab_size: int = VOCAB_SIZE,
    ) -> "HydraTransformerConfig":
        """Factory for OLMo2 7B (32 layers) with configurable split point."""
        from olmo_core.nn.attention import AttentionBackendName

        base = TransformerConfig.olmo2_7B(vocab_size=vocab_size)
        base.block.sequence_mixer.backend = AttentionBackendName.flash_2  # type: ignore[union-attr]
        return cls(
            base_config=base,
            n_heads=n_heads,
            trunk_layers=base.n_layers - heads_depth,
            head_layers=heads_depth,
        )

    def build(self, *, init_device: str = "cpu") -> "HydraTransformer":
        """
        Build the HydraTransformer.

        :param init_device: Device for parameter initialization (use ``"meta"`` for zero-memory).
        """
        # new configs for the trunk and heads.
        trunk_config = replace(self.base_config, n_layers=self.trunk_layers)
        head_config = replace(self.base_config, n_layers=self.head_layers)

        # build meta trunk, no need for lm_head
        trunk = trunk_config.build(init_device=init_device)
        trunk.lm_head = None  # type: ignore[assignment]

        # Build one head to extract the shared lm_head, then strip it
        donor = head_config.build(init_device=init_device)
        lm_head = donor.lm_head
        donor.lm_head = None  # type: ignore[assignment]
        donor.embeddings = None  # type: ignore[assignment]

        # build all the meta heads, all without lm_head.
        # NOTE: currently only use one lm_head shared for all hydra heads.
        # think LoRA fine tuning usually does not touch this.
        heads = nn.ModuleList()
        heads.append(donor)
        for _ in range(self.n_heads - 1):
            head = head_config.build(init_device=init_device)
            head.embeddings = None  # type: ignore[assignment]
            head.lm_head = None  # type: ignore[assignment]
            heads.append(head)

        return HydraTransformer(trunk=trunk, heads=heads, lm_head=lm_head)

    @property
    def num_params(self) -> int:
        assert not isinstance(self.base_config.block, dict)
        d = self.base_config.d_model
        block_params = self.base_config.block.num_params(d)

        # Trunk: embeddings + trunk blocks (no lm_head).
        n = d * self.base_config.vocab_size + self.trunk_layers * block_params

        # Heads: head blocks only (no embeddings, no lm_head).
        n += self.n_heads * self.head_layers * block_params

        # Shared lm_head: one copy.
        n += self.base_config.lm_head.num_params(d, self.base_config.vocab_size)

        return n

    @property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.base_config.d_model * self.base_config.vocab_size


class HydraTransformer(nn.Module):
    """
    A multi-head branched transformer.

    Runs input through a shared trunk, then fans out to N independent heads.
    All heads share a single lm_head for the final projection to vocab logits.

    :param trunk: Shared transformer trunk (no lm_head).
    :param heads: ModuleList of head transformers (no embeddings, no lm_head).
    :param lm_head: Shared language modeling head.
    """

    def __init__(
        self,
        trunk: Transformer,
        heads: nn.ModuleList,
        lm_head: nn.Module,
    ):
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.lm_head = lm_head

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_kv_cache(self, batch_size: int, max_seq_len: int):
        """Initialize KV caches for all blocks in trunk and heads."""
        for block in self.trunk.blocks.values():
            attn = cast(TransformerBlock, block).attention
            if isinstance(attn, Attention):
                attn.init_kv_cache_manager(batch_size, max_seq_len)
        for head in self.heads:
            for block in cast(Transformer, head).blocks.values():
                attn = cast(TransformerBlock, block).attention
                if isinstance(attn, Attention):
                    attn.init_kv_cache_manager(batch_size, max_seq_len)

    def reset_kv_cache(self):
        """Reset KV cache position counters to 0 before each generation."""
        for block in self.trunk.blocks.values():
            attn = cast(TransformerBlock, block).attention
            if isinstance(attn, Attention) and attn.kv_cache_manager is not None:
                attn.kv_cache_manager.cache_seqlens.fill_(0)
        for head in self.heads:
            for block in cast(Transformer, head).blocks.values():
                attn = cast(TransformerBlock, block).attention
                if isinstance(attn, Attention) and attn.kv_cache_manager is not None:
                    attn.kv_cache_manager.cache_seqlens.fill_(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        residual: torch.Tensor | None = None,
        head_indices: list[int] | None = None,
        last_token_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the full model.

        :param input_ids: Token IDs ``(batch, seq)``.
        :param head_indices: Optional subset of head indices to run. None means all heads.
        :param last_token_only: If True, project only the final sequence position through
            the lm_head. Output seq dim collapses to 1. Cheap inference path for
            classification / next-token argmax; training must keep False.
        :returns: Logits tensor ``(n_selected, batch, seq_out, vocab)`` where
            ``seq_out == 1`` if ``last_token_only`` else ``seq``.
        """
        h = self.trunk(input_ids, **kwargs)

        if residual is not None:
            assert residual.size == h.size, (
                f"Residual shape mismatch, expected {h.size} got {residual.size}"
            )
            h += residual

        if head_indices is not None:
            if len(head_indices) == 0:
                raise ValueError("head_indices must be non-empty")
            n = len(self.heads)
            for idx in head_indices:
                if idx < 0 or idx >= n:
                    raise ValueError(f"head index {idx} out of range for {n} heads")
            selected = [self.heads[i] for i in head_indices]
        else:
            selected = list(self.heads)

        head_hidden = [head(h, **kwargs) for head in selected]

        stacked = torch.cat(head_hidden, dim=0)
        if last_token_only:
            stacked = stacked[:, -1:, :]
        all_logits: torch.Tensor = self.lm_head(stacked)
        return all_logits.unflatten(0, (len(selected), -1))

    def residual_forward(
        self, input_ids: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full model and return variance of head final layer hidden states.
        """
        h = self.trunk(input_ids, **kwargs)

        head_hidden = [head(h, **kwargs) for head in self.heads]
        stacked = torch.stack(head_hidden)  # (N, batch, seq, d_model)
        hidden_var = stacked.var(dim=0)  # (batch, seq, d_model)

        # Flatten to (N*batch, seq, vocab) for lm_head, then unflatten back to (N, batch, seq, vocab)
        all_logits: torch.Tensor = self.lm_head(stacked.flatten(0, 1))
        return all_logits.unflatten(0, (len(self.heads), -1)), hidden_var

    @staticmethod
    def load_olmo_state(
        model: "HydraTransformer",
        olmo_state: dict[str, torch.Tensor],
        trunk_layers: int,
        vocab_size: int,
    ) -> None:
        """
        Load a flat OLMo-format state dict into a HydraTransformer.

        Splits the state by layer index into trunk/head/lm_head components,
        pads vocab embeddings if needed, and clones head weights for each head.

        :param model: The HydraTransformer to load into.
        :param olmo_state: OLMo-format state dict (output of ``convert_state_from_hf``).
        :param trunk_layers: Number of layers in the trunk.
        :param vocab_size: Target vocab size (for padding).
        """
        trunk_state: dict[str, torch.Tensor] = {}
        head_state: dict[str, torch.Tensor] = {}
        lm_head_state: dict[str, torch.Tensor] = {}

        for key, value in olmo_state.items():
            if key.startswith("blocks."):
                block_idx = int(key.split(".", 2)[1])
                suffix = key.split(".", 2)[2]
                if block_idx < trunk_layers:
                    trunk_state[f"blocks.{block_idx}.{suffix}"] = value
                else:
                    new_idx = block_idx - trunk_layers
                    head_state[f"blocks.{new_idx}.{suffix}"] = value
            elif key.startswith("lm_head."):
                lm_head_state[key.split(".", 1)[1]] = value
            else:
                trunk_state[key] = value

        # pad vocab so that it is a nice size for matmuls
        emb = trunk_state["embeddings.weight"]
        if emb.shape[0] < vocab_size:
            padding = torch.zeros(
                vocab_size - emb.shape[0], emb.shape[1], dtype=emb.dtype
            )
            trunk_state["embeddings.weight"] = torch.cat([emb, padding], dim=0)

        w_out = lm_head_state["w_out.weight"]
        if w_out.shape[0] < vocab_size:
            padding = torch.zeros(
                vocab_size - w_out.shape[0], w_out.shape[1], dtype=w_out.dtype
            )
            lm_head_state["w_out.weight"] = torch.cat([w_out, padding], dim=0)

        model.trunk.load_state_dict(trunk_state, assign=True)
        model.lm_head.load_state_dict(lm_head_state, assign=True)

        for i, head in enumerate(model.heads):
            # NOTE: For testing, can inject noise into head params here
            state = (
                head_state if i == 0 else {k: v.clone() for k, v in head_state.items()}
            )  # NEED COPY
            head.load_state_dict(state, assign=True)
