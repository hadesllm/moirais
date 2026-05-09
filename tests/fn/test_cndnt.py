"""Tests for conditional entropy."""
import numpy as np
import pytest
from moirais.fn.cndnt import conditional_entropy, cndnt


def test_nonnegative():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 1, 5000)
    r = conditional_entropy(x, y, bins=8)
    assert r.estimate >= -0.1


def test_alias():
    assert cndnt is conditional_entropy


def test_length_mismatch():
    with pytest.raises(ValueError):
        conditional_entropy([1, 2], [1])
