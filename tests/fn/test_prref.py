"""Tests for moirais.fn.prref -- Procrustes reflection."""

import numpy as np
from moirais.fn.prref import procrustes_reflection, prref


def test_prref_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 2))
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Y = X @ R.T
    r = prref(X, Y)
    assert r.name == "procrustes_reflection"
    assert r.value.shape == (5, 2)
    assert r.extra["disparity"] < 1e-8


def test_prref_alias():
    assert prref is procrustes_reflection
