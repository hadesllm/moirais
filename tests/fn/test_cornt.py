"""Tests for correlation entropy."""
import numpy as np
from moirais.fn.cornt import correlation_entropy, cornt


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    r = correlation_entropy(x, m_max=5)
    assert r.estimate >= 0


def test_alias():
    assert cornt is correlation_entropy
