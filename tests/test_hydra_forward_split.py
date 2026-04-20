import pytest
import torch

from olmo_tap.hydra import HydraTransformerConfig


@pytest.fixture
def tiny_model():
    """Tiny HydraTransformer on CUDA with flash-2 attention."""
    config = HydraTransformerConfig.from_olmo2_1B(n_heads=3, heads_depth=2)
    m = config.build(init_device="cuda")
    m.to(dtype=torch.bfloat16)
    m.eval()
    return m


@pytest.fixture
def tiny_input():
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 8), device="cuda")


def test_tiny_model_forward_smoke(tiny_model, tiny_input):
    with torch.no_grad():
        out = tiny_model(tiny_input)
    # (n_heads, batch, seq, vocab)
    assert out.ndim == 4
    assert out.shape[0] == 3
    assert out.shape[1] == 1
    assert out.shape[2] == 8


def test_forward_trunk_returns_hidden_states(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
    # (batch, seq, d_model)
    assert h.ndim == 3
    assert h.shape[0] == 1
    assert h.shape[1] == 8


def test_forward_heads_default_runs_all(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
        out = tiny_model.forward_heads(h)
    assert out.shape[0] == 3
    assert out.shape[1] == 1
    assert out.shape[2] == 8


def test_forward_heads_respects_head_indices(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
        out = tiny_model.forward_heads(h, head_indices=[0, 2])
    assert out.shape[0] == 2


def test_forward_heads_last_token_only(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
        out = tiny_model.forward_heads(h, last_token_only=True)
    assert out.shape[2] == 1


def test_forward_heads_rejects_empty_indices(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
    with pytest.raises(ValueError, match="non-empty"):
        tiny_model.forward_heads(h, head_indices=[])


def test_forward_heads_rejects_out_of_range(tiny_model, tiny_input):
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
    with pytest.raises(ValueError, match="out of range"):
        tiny_model.forward_heads(h, head_indices=[3])


def test_forward_matches_trunk_then_heads(tiny_model, tiny_input):
    """After the refactor, forward(x) must equal forward_heads(forward_trunk(x))."""
    with torch.no_grad():
        expected = tiny_model(tiny_input)
        h = tiny_model.forward_trunk(tiny_input)
        actual = tiny_model.forward_heads(h)
    torch.testing.assert_close(actual, expected)


def test_forward_matches_split_with_head_indices(tiny_model, tiny_input):
    with torch.no_grad():
        expected = tiny_model(tiny_input, head_indices=[1])
        h = tiny_model.forward_trunk(tiny_input)
        actual = tiny_model.forward_heads(h, head_indices=[1])
    torch.testing.assert_close(actual, expected)


def test_forward_matches_split_last_token_only(tiny_model, tiny_input):
    with torch.no_grad():
        expected = tiny_model(tiny_input, last_token_only=True)
        h = tiny_model.forward_trunk(tiny_input)
        actual = tiny_model.forward_heads(h, last_token_only=True)
    torch.testing.assert_close(actual, expected)


def test_residual_forward_still_returns_logits_and_variance(tiny_model, tiny_input):
    with torch.no_grad():
        logits, var = tiny_model.residual_forward(tiny_input)
    # (n_heads, batch, seq, vocab)
    assert logits.shape[0] == 3
    # (batch, seq, d_model)
    assert var.ndim == 3
    assert var.shape[0] == 1
    assert var.shape[1] == 8


