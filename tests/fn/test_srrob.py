"""Test srrob."""
import numpy as np
import pytest
from morie.fn.srrob import srrob


def test_srrob_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    y = X @ np.array([1, -0.5, 0.3]) + rng.standard_normal(30) * 0.5
    r = srrob(X=X, y=y, n=30, k=3)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert 0 < r.value <= 1.0
    assert r.extra["r_squared"] == pytest.approx(r.value, rel=1e-10)


def test_srrob_description():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    y = X @ np.array([1, -0.5, 0.3]) + rng.standard_normal(30) * 0.5
    r = srrob(X=X, y=y, n=30, k=3)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert len(r.extra["coefficients"]) == 3
