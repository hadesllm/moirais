"""Tests for joint entropy."""
import numpy as np
import pytest
from morie.fn.jntnt import joint_entropy, jntnt


def test_independent():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 1, 5000)
    r = joint_entropy(x, y, bins=8)
    assert r.estimate > 0


def test_identical():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 5000)
    r = joint_entropy(x, x, bins=8)
    assert r.estimate > 0


def test_alias():
    assert jntnt is joint_entropy


def test_length_mismatch():
    with pytest.raises(ValueError):
        joint_entropy([1, 2], [1, 2, 3])
