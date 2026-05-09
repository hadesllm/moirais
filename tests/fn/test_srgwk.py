"""Test srgwk."""
import numpy as np
import pytest
from moirais.fn.srgwk import srgwk


def test_srgwk_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    y = X @ np.array([1, -0.5, 0.3]) + rng.standard_normal(30) * 0.5
    r = srgwk(X=X, y=y, n=30, k=3)
    assert r.value is not None


def test_srgwk_description():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    y = X @ np.array([1, -0.5, 0.3]) + rng.standard_normal(30) * 0.5
    r = srgwk(X=X, y=y, n=30, k=3)
    assert r.name
