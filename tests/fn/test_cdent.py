"""Tests for conditional_entropy."""
import numpy as np
import pytest
from morie.fn.cdent import conditional_entropy, cdent


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    y = x + rng.normal(0, 0.1, 1000)
    r = conditional_entropy(x, y, bins=20)
    assert r.estimate >= 0


def test_alias():
    assert cdent is conditional_entropy


def test_length_mismatch():
    with pytest.raises(ValueError):
        conditional_entropy([1, 2], [1])
