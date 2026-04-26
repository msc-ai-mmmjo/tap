"""Entrant definitions for the configuration-level Elo tournament.

Defines the four configurations being compared, a dispatch table for the
generation harness, and the loader that materialises each entrant on
GPU. The :class:`EntrantSpec` instances are deliberately serialisable
(no torch / model objects) so the full tournament configuration can be
hashed for cache keys and logged in the run manifest without GPU
dependencies; the loaded resources live in :class:`LoadedEntrant`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from olmo_tap.constants import WEIGHTS_DIR
from olmo_tap.final_evals._loading import load_eval_hydra
from olmo_tap.hydra import HydraTransformer
from olmo_tap.inference.poe import PoE


EntrantId = Literal[
    "base_olmo",
    "security_only",
    "security_plus_robustness",
    "full_poe",
]

LoaderName = Literal["vanilla_hf", "custom_poe"]


@dataclass(frozen=True)
class EntrantSpec:
    """Structural description of a tournament entrant.

    The fields are deliberately serialisable (no torch / model objects) so the
    full tournament configuration can be hashed for cache keys and logged in
    the run manifest without GPU dependencies.

    Attributes:
        entrant_id: Stable identifier used in caches, manifests, and reports.
        loader: Which model-loading recipe to use. ``vanilla_hf`` is the
            plain HuggingFace ``AutoModelForCausalLM`` path used by the base
            OLMo entrant. ``custom_poe`` mirrors :func:`load_custom_poe` from
            ``robustness_sweep.py`` for the Hydra entrants.
        rob_checkpoint: Robustness LoRA checkpoint to merge. ``None`` skips
            the robustness merge entirely (security-only entrant). ``-1``
            selects the final checkpoint. Otherwise a step index.
        bypass_jury: When ``True`` the PoE jury is short-circuited and every
            draft token is accepted. Used to get a single head's sampled
            output through the PoE codepath. Set ``False`` for full
            speculative decoding with verifier acceptance.
        temperature: Sampling temperature. ``None`` selects greedy decoding
            (the base OLMo benchmark convention). Hydra entrants use
            ``0.98`` to match the production PoE path.
        needs_uncertainty: When ``True`` PoE captures the witness hidden
            state and computes ``p_correct`` via the second-pass uncertainty
            head. The configuration-level Elo report does not consume
            ``p_correct`` directly so all four entrants set this to
            ``False`` for now.
    """

    entrant_id: EntrantId
    loader: LoaderName
    rob_checkpoint: int | None
    bypass_jury: bool
    temperature: float | None
    needs_uncertainty: bool = False


@dataclass
class LoadedEntrant:
    """A materialised entrant: spec + the model resources needed to generate.

    Attributes:
        spec: The :class:`EntrantSpec` this loader instance was built from.
        tokenizer: Tokenizer paired with the model.
        hf_model: Vanilla HuggingFace causal-LM, populated only when
            ``spec.loader == "vanilla_hf"``. ``None`` otherwise.
        hydra: Underlying :class:`HydraTransformer`, populated only when
            ``spec.loader == "custom_poe"``. ``None`` otherwise.
        poe: :class:`PoE` wrapping ``hydra``, populated only when
            ``spec.loader == "custom_poe"``. ``None`` otherwise.
    """

    spec: EntrantSpec
    tokenizer: PreTrainedTokenizerBase
    hf_model: Optional[PreTrainedModel] = None
    hydra: Optional[HydraTransformer] = None
    poe: Optional[PoE] = None


ENTRANTS: tuple[EntrantSpec, ...] = (
    EntrantSpec(
        entrant_id="base_olmo",
        loader="vanilla_hf",
        rob_checkpoint=None,
        bypass_jury=True,
        temperature=None,
        needs_uncertainty=False,
    ),
    EntrantSpec(
        entrant_id="security_only",
        loader="custom_poe",
        rob_checkpoint=None,
        bypass_jury=True,
        temperature=0.98,
        needs_uncertainty=False,
    ),
    EntrantSpec(
        entrant_id="security_plus_robustness",
        loader="custom_poe",
        rob_checkpoint=-1,
        bypass_jury=True,
        temperature=0.98,
        needs_uncertainty=False,
    ),
    EntrantSpec(
        entrant_id="full_poe",
        loader="custom_poe",
        rob_checkpoint=-1,
        bypass_jury=False,
        temperature=0.98,
        needs_uncertainty=False,
    ),
)


ENTRANTS_BY_ID: dict[str, EntrantSpec] = {e.entrant_id: e for e in ENTRANTS}


def get_entrant(entrant_id: str) -> EntrantSpec:
    """Look up an :class:`EntrantSpec` by its stable id."""
    try:
        return ENTRANTS_BY_ID[entrant_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown entrant_id {entrant_id!r}. Known: {sorted(ENTRANTS_BY_ID)}"
        ) from exc


def build_entrant(spec: EntrantSpec, max_new_tokens: int = 256) -> LoadedEntrant:
    """Materialise the model + tokenizer pair for an entrant.

    Both loader paths land their tensors on CUDA in bfloat16 so generation
    is comparable across entrants. The ``custom_poe`` path additionally
    wraps the loaded :class:`HydraTransformer` in :class:`PoE` so the
    eval harness can call ``generate_with_cache`` uniformly.
    """
    if spec.loader == "vanilla_hf":
        hf_model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                WEIGHTS_DIR, torch_dtype=torch.bfloat16
            ).to("cuda"),
        )
        hf_model.eval()
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
        )
        return LoadedEntrant(spec=spec, tokenizer=tokenizer, hf_model=hf_model)

    if spec.loader == "custom_poe":
        merge_robustness = spec.rob_checkpoint is not None
        hydra, n_heads = load_eval_hydra(merge_robustness=merge_robustness)
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
        )
        poe = PoE(
            hydra,
            tokenizer,
            n_llm_heads=n_heads - 1,
            max_new_tokens=max_new_tokens,
        )
        return LoadedEntrant(spec=spec, tokenizer=tokenizer, hydra=hydra, poe=poe)

    raise ValueError(f"Unknown loader {spec.loader!r} for entrant {spec.entrant_id}")
