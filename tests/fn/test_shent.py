"""Tests for shannon_entropy."""
import numpy as np
from morie.fn.shent import shannon_entropy, shent


def test_uniform():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 10000)
    r = shannon_entropy(x, bins=8)
    assert abs(r.estimate - 3.0) < 0.2


def test_alias():
    assert shent is shannon_entropy


def test_nats():
    r = shannon_entropy([1, 2, 3, 4], bins=4)
    assert r.extra["nats"] > 0
