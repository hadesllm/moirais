"""Tests for Tsallis entropy."""
import numpy as np
import pytest
from morie.fn.tsent import tsallis_entropy, tsent


def test_positive():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 10000)
    r = tsallis_entropy(x, q=2.0, bins=8)
    assert r.estimate > 0


def test_q_one_raises():
    with pytest.raises(ValueError):
        tsallis_entropy([1, 2, 3], q=1.0)


def test_alias():
    assert tsent is tsallis_entropy
