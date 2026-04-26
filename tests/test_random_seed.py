import random

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from olmo_tap.experiments.utils.random_seed import set_seed


def _draw():
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def test_set_seed_is_deterministic_across_calls():
    set_seed(42)
    first = _draw()
    set_seed(42)
    second = _draw()
    assert first == second


def test_set_seed_distinct_seeds_diverge():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_set_seed_does_not_raise_without_cuda():
    # The function calls torch.cuda.manual_seed unconditionally; on a CPU-only
    # box this must remain a no-op rather than raising.
    set_seed(0)
