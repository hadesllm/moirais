"""Tests for morie.fn.dimts -- dimensionality test."""
import numpy as np
from morie.fn.dimts import dimensionality_test, dimts


def test_alias():
    assert dimts is dimensionality_test


def test_smoke():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((50, 5))
    r = dimensionality_test(data, max_dims=3)
    assert r.name == "dimensionality_test"
    assert len(r.extra["test_stats"]) == 3
    assert r.extra["test_stats"][-1]["explained_var"] > 0


def test_eigenvalues():
    data = np.random.default_rng(42).standard_normal((30, 4))
    r = dimensionality_test(data, max_dims=4)
    assert len(r.extra["eigenvalues"]) == 4
    assert r.extra["eigenvalues"][0] >= r.extra["eigenvalues"][1]
