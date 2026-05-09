"""Test cancc."""
import numpy as np
import pytest
from moirais.fn.cancc import canonical_correlation


def test_cancc_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    Y = rng.standard_normal((100, 2))
    r = canonical_correlation(X, Y)
    assert 0.0 <= r.value <= 1.0
    assert r.name == "cancc"


def test_cancc_correlated():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 2))
    Y = X + rng.standard_normal((80, 2)) * 0.1
    r = canonical_correlation(X, Y)
    assert r.value > 0.5
    assert len(r.extra["canonical_correlations"]) == 2
