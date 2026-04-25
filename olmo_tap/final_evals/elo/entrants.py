"""Entrant definitions for the configuration-level Elo tournament.

Defines the four configurations being compared and a dispatch table that the
generation harness uses to materialise each entrant. The actual model loading
recipes are not yet wired up — :func:`build_entrant` is intentionally
stubbed for now and will land alongside the response-generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def build_entrant(spec: EntrantSpec):
    """Materialise the model + tokenizer pair for an entrant.

    Not yet implemented — actual loading lands once the eval-mode kwargs on
    ``PoE.generate_with_cache`` (per-prompt seeding and ``bypass_jury``)
    have been added. The dispatch will branch on ``spec.loader`` and reuse:

    - ``vanilla_hf``: ``AutoModelForCausalLM.from_pretrained`` + greedy
      ``model.generate``.
    - ``custom_poe``: the body of ``load_custom_poe()`` from
      ``robustness_sweep.py`` wrapped in :class:`PoE`.
    """
    raise NotImplementedError(
        "build_entrant is not yet wired up — pending the eval-mode kwargs "
        "on PoE.generate_with_cache."
    )
