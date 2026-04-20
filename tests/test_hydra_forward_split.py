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


def _all_cache_seqlens(model):
    """Return cache_seqlens for trunk blocks followed by each head's blocks."""
    from olmo_core.nn.attention import Attention

    out = []
    for block in model.trunk.blocks.values():
        attn = block.attention
        if isinstance(attn, Attention) and attn.kv_cache_manager is not None:
            out.append(int(attn.kv_cache_manager.cache_seqlens.item()))
    for head in model.heads:
        for block in head.blocks.values():
            attn = block.attention
            if isinstance(attn, Attention) and attn.kv_cache_manager is not None:
                out.append(int(attn.kv_cache_manager.cache_seqlens.item()))
    return out


def _head_cache_seqlens(head):
    from olmo_core.nn.attention import Attention

    return [
        int(block.attention.kv_cache_manager.cache_seqlens.item())
        for block in head.blocks.values()
        if isinstance(block.attention, Attention)
        and block.attention.kv_cache_manager is not None
    ]


def _trunk_cache_seqlens(model):
    from olmo_core.nn.attention import Attention

    return [
        int(block.attention.kv_cache_manager.cache_seqlens.item())
        for block in model.trunk.blocks.values()
        if isinstance(block.attention, Attention)
        and block.attention.kv_cache_manager is not None
    ]


def test_rollback_decrements_all_pointers(tiny_model, tiny_input):
    tiny_model.init_kv_cache(batch_size=1, max_seq_len=16)
    with torch.no_grad():
        tiny_model(tiny_input)
    before = _all_cache_seqlens(tiny_model)
    assert all(v == 8 for v in before), (
        f"expected all cache pointers at 8, got {before}"
    )

    tiny_model.rollback_kv_cache(3)
    after = _all_cache_seqlens(tiny_model)
    assert all(v == 5 for v in after), f"expected all cache pointers at 5, got {after}"


def test_rollback_zero_is_noop(tiny_model, tiny_input):
    """rollback_kv_cache(0) should leave every pointer unchanged."""
    tiny_model.init_kv_cache(batch_size=1, max_seq_len=16)
    with torch.no_grad():
        tiny_model(tiny_input)
    before = _all_cache_seqlens(tiny_model)
    tiny_model.rollback_kv_cache(0)
    after = _all_cache_seqlens(tiny_model)
    assert before == after


def test_forward_trunk_does_not_advance_head_caches(tiny_model, tiny_input):
    tiny_model.init_kv_cache(batch_size=1, max_seq_len=16)
    with torch.no_grad():
        _ = tiny_model.forward_trunk(tiny_input)
    trunk_positions = _trunk_cache_seqlens(tiny_model)
    head_positions = [p for head in tiny_model.heads for p in _head_cache_seqlens(head)]
    assert all(p == 8 for p in trunk_positions), (
        f"trunk not advanced: {trunk_positions}"
    )
    assert all(p == 0 for p in head_positions), (
        f"heads should not have advanced: {head_positions}"
    )


def test_forward_heads_advances_only_selected(tiny_model, tiny_input):
    tiny_model.init_kv_cache(batch_size=1, max_seq_len=16)
    with torch.no_grad():
        h = tiny_model.forward_trunk(tiny_input)
        trunk_before = _trunk_cache_seqlens(tiny_model)
        _ = tiny_model.forward_heads(h, head_indices=[1])
    trunk_after = _trunk_cache_seqlens(tiny_model)
    assert trunk_after == trunk_before, (
        f"trunk advanced during forward_heads: {trunk_before} -> {trunk_after}"
    )
    assert all(p == 0 for p in _head_cache_seqlens(tiny_model.heads[0]))
    assert all(p == 8 for p in _head_cache_seqlens(tiny_model.heads[1]))
    assert all(p == 0 for p in _head_cache_seqlens(tiny_model.heads[2]))


def test_rollback_then_refill_matches_direct_forward(tiny_model):
    """Forward x1+x2, rollback len(x2), forward x2 again. Last-token logits should match."""
    torch.manual_seed(42)
    x1 = torch.randint(0, 100, (1, 6), device="cuda")
    x2 = torch.randint(0, 100, (1, 3), device="cuda")

    tiny_model.init_kv_cache(batch_size=1, max_seq_len=32)
    with torch.no_grad():
        full = torch.cat([x1, x2], dim=1)
        reference_logits = tiny_model(full)[:, :, -1, :].clone()

    tiny_model.reset_kv_cache()
    with torch.no_grad():
        _ = tiny_model(x1)
        _ = tiny_model(x2)
        tiny_model.rollback_kv_cache(3)
        actual_logits = tiny_model(x2)[:, :, -1, :].clone()

    torch.testing.assert_close(actual_logits, reference_logits, rtol=1e-2, atol=1e-2)
