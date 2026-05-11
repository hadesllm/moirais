"""Tests for SVD entropy."""
import numpy as np
from morie.fn.svdnt import svd_entropy, svdnt


def test_normalised():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    r = svd_entropy(x, m=5)
    assert 0.0 <= r.estimate <= 1.0


def test_alias():
    assert svdnt is svd_entropy
