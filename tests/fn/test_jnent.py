"""Tests for joint_entropy."""
import numpy as np
import pytest
from moirais.fn.jnent import joint_entropy, jnent


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0, 1, 1000)
    r = joint_entropy(x, y, bins=10)
    assert r.estimate > 0


def test_alias():
    assert jnent is joint_entropy


def test_length_mismatch():
    with pytest.raises(ValueError):
        joint_entropy([1, 2], [1])
