"""Tests for transfer entropy."""
import numpy as np
from moirais.fn.trsfn import transfer_entropy, trsfn


def test_independent():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    r = transfer_entropy(x, y, k=1, bins=10)
    assert abs(r.estimate) < 1.0


def test_causal():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 500)
    x = np.zeros(500)
    x[1:] = 0.8 * y[:-1] + 0.2 * rng.normal(0, 1, 499)
    r = transfer_entropy(x, y, k=1, bins=10)
    assert r.estimate > 0


def test_alias():
    assert trsfn is transfer_entropy
