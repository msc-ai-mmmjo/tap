import math

import pytest

from kernel_entropy.entropy import kle_to_certainty


def test_kle_to_certainty_zero_entropy_is_full_certainty():
    assert kle_to_certainty(0.0, 5) == 1.0


def test_kle_to_certainty_max_entropy_is_zero_certainty():
    assert kle_to_certainty(math.log(5), 5) == pytest.approx(0.0, abs=1e-9)


def test_kle_to_certainty_midpoint():
    assert kle_to_certainty(math.log(5) / 2, 5) == pytest.approx(0.5, abs=1e-9)


def test_kle_to_certainty_requires_two_samples():
    assert kle_to_certainty(0.0, 1) is None


def test_kle_to_certainty_clamps_overflow_to_zero():
    # Slight numerical noise above log(N) must not produce a negative certainty.
    over = math.log(5) + 1e-6
    assert kle_to_certainty(over, 5) == 0.0
