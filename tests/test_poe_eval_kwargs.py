"""Unit tests for the eval-mode kwargs added to ``PoE.generate_with_cache``.

The tests use a small mock model and tokenizer rather than loading OLMo2-7B —
they exercise the control flow added by ``compute_uncertainty``, ``seed``,
and ``bypass_jury`` plus the ``last_diagnostics`` snapshot. CUDA is required
because :class:`~olmo_tap.inference.poe.PoE` allocates intermediate tensors
on ``"cuda"`` directly; tests are skipped otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from olmo_tap.inference.poe import PoE


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for PoE tests"
)


VOCAB_SIZE = 32
N_LLM_HEADS = 4
D_MODEL = 8
PROMPT_LEN = 5
GAMMA = 4
EOS_ID = VOCAB_SIZE - 1
A_ID = 1
B_ID = 2


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()

    def encode(text, add_special_tokens=True):
        if text == "A":
            return [A_ID]
        if text == "B":
            return [B_ID]
        return list(range(3, 3 + PROMPT_LEN))

    tok.encode.side_effect = encode
    tok.apply_chat_template.return_value = "<chat>"

    def decode(ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "<" + ",".join(str(int(x)) for x in ids) + ">"

    tok.decode.side_effect = decode
    tok.eos_token_id = EOS_ID
    return tok


def _logits_no_eos(*shape: int) -> torch.Tensor:
    """Random logits with EOS suppressed so generation never short-circuits."""
    out = torch.randn(*shape, device="cuda")
    out[..., EOS_ID] = -1e9
    return out


def _make_model() -> MagicMock:
    """A mock HydraTransformer with the surface PoE actually consumes."""
    model = MagicMock()

    model.init_kv_cache.return_value = None
    model.sync_kv_cache.return_value = None

    def residual_forward(input_ids, last_token_only, head_indices, hidden_head_indices):
        n = len(head_indices)
        next_step_logits = _logits_no_eos(n, 1, 1, VOCAB_SIZE)
        hidden_bank = torch.randn(n, 1, input_ids.size(1), D_MODEL, device="cuda")
        return next_step_logits, hidden_bank

    model.residual_forward.side_effect = residual_forward

    def forward_trunk(curr_d_token):
        return torch.randn(1, 1, D_MODEL, device="cuda")

    model.forward_trunk.side_effect = forward_trunk

    def forward_heads(h, head_indices):
        n = len(head_indices)
        seq_len = h.size(1)
        return _logits_no_eos(n, 1, seq_len, VOCAB_SIZE)

    model.forward_heads.side_effect = forward_heads

    # Models __call__ used on rejection path; not exercised by these tests but
    # we still wire a sensible return so any accidental invocation is debuggable.
    model.return_value = _logits_no_eos(N_LLM_HEADS, 1, 1, VOCAB_SIZE)

    return model


def _make_poe(max_new_tokens: int = GAMMA) -> PoE:
    model = _make_model()
    tokenizer = _make_tokenizer()
    poe = PoE(
        model,
        tokenizer,
        n_llm_heads=N_LLM_HEADS,
        gamma=GAMMA,
        max_new_tokens=max_new_tokens,
    )
    return poe


def test_default_kwargs_run_full_jury_path():
    """With no eval-mode kwargs set, the verifier forward must be invoked.

    Backward-compat smoke: if any future change accidentally short-circuits
    the verify block at default args, this fails.
    """
    poe = _make_poe()
    poe.generate_with_cache("hello", temperature=None)

    head_indices_seen = [
        call.kwargs.get("head_indices", call.args[1] if len(call.args) > 1 else None)
        for call in poe.model.forward_heads.call_args_list  # type: ignore[attr-defined]
    ]
    # The verify step calls forward_heads with the verifier indices (size > 1
    # since N_LLM_HEADS - 1 verifiers). Default path must reach it.
    assert any(
        isinstance(idxs, list) and len(idxs) > 1 for idxs in head_indices_seen
    ), f"verifier forward never invoked under default kwargs: {head_indices_seen}"


def test_bypass_jury_skips_verifier_forward():
    poe = _make_poe()
    poe.generate_with_cache("hello", bypass_jury=True, temperature=None)

    draft_idx = poe.last_diagnostics["draft_head_idx"]
    head_indices_seen = [
        call.kwargs.get("head_indices", call.args[1] if len(call.args) > 1 else None)
        for call in poe.model.forward_heads.call_args_list  # type: ignore[attr-defined]
    ]
    # Every forward_heads call should target only the draft head — no verifier
    # forward, ever.
    for idxs in head_indices_seen:
        assert idxs == [draft_idx], (
            f"forward_heads called with {idxs} under bypass_jury (expected only [{draft_idx}])"
        )


def test_bypass_jury_zero_resampled_in_diagnostics():
    poe = _make_poe()
    poe.generate_with_cache("hello", bypass_jury=True, temperature=None)
    assert poe.last_diagnostics["bypass_jury"] is True
    assert poe.last_diagnostics["n_resampled"] == 0


def test_seed_determinism_across_calls():
    """Two calls with the same seed must select the same draft head."""
    poe1 = _make_poe()
    poe2 = _make_poe()

    poe1.generate_with_cache("hello", seed=12345, bypass_jury=True, temperature=None)
    poe2.generate_with_cache("hello", seed=12345, bypass_jury=True, temperature=None)

    assert (
        poe1.last_diagnostics["draft_head_idx"]
        == poe2.last_diagnostics["draft_head_idx"]
    )
    assert poe1.last_diagnostics["seed"] == 12345


def test_compute_uncertainty_true_with_is_mcq_false_invokes_second_pass():
    """Forcing compute_uncertainty=True on a non-MCQ prompt must capture the
    witness state and run the uncertainty second pass."""
    poe = _make_poe()
    poe.get_uncertainty_score = MagicMock(return_value=0.42)  # type: ignore[method-assign]

    _, _, _, score = poe.generate_with_cache(
        "hello",
        is_mcq=False,
        compute_uncertainty=True,
        bypass_jury=True,
        temperature=None,
    )

    poe.get_uncertainty_score.assert_called_once()
    assert score == 0.42


def test_compute_uncertainty_false_with_is_mcq_true_skips_second_pass():
    """compute_uncertainty=False overrides is_mcq=True and must skip capture."""
    poe = _make_poe()
    poe.get_uncertainty_score = MagicMock(return_value=0.42)  # type: ignore[method-assign]

    _, _, _, score = poe.generate_with_cache(
        "hello",
        is_mcq=True,
        compute_uncertainty=False,
        bypass_jury=True,
        temperature=None,
    )

    poe.get_uncertainty_score.assert_not_called()
    assert score is None


def test_last_diagnostics_populated_under_default_kwargs():
    poe = _make_poe()
    poe.generate_with_cache("hello", temperature=None)

    diag = poe.last_diagnostics
    assert set(diag.keys()) == {
        "draft_head_idx",
        "seed",
        "bypass_jury",
        "n_resampled",
        "n_tokens_generated",
    }
    assert 0 <= diag["draft_head_idx"] < N_LLM_HEADS
    assert diag["seed"] is None
    assert diag["bypass_jury"] is False
    assert isinstance(diag["n_resampled"], int)
    assert diag["n_tokens_generated"] >= 1


def test_last_diagnostics_records_seed_and_bypass():
    poe = _make_poe()
    poe.generate_with_cache("hello", seed=7, bypass_jury=True, temperature=None)
    diag = poe.last_diagnostics
    assert diag["seed"] == 7
    assert diag["bypass_jury"] is True


def test_last_diagnostics_initialised_in_init():
    """Attribute exists immediately after construction even before any call."""
    poe = _make_poe()
    assert poe.last_diagnostics == {}
