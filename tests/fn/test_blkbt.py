"""Tests for moirais.fn.blkbt — Blackbox scaling."""
import numpy as np
from moirais.fn.blkbt import blkbt


def test_blkbt_basic():
    R = np.array([[1, 5, 3], [5, 1, 4], [2, 4, 3], [4, 2, 5]], dtype=float)
    r = blkbt(R, n_dims=1)
    assert r.value["ideal_points"].shape == (4, 1)


def test_blkbt_explained():
    rng = np.random.default_rng(42)
    R = rng.integers(1, 6, (10, 5)).astype(float)
    r = blkbt(R, n_dims=2)
    assert 0 < r.value["explained_variance"] <= 1


def test_blkbt_extra():
    R = np.array([[1, 2], [2, 1]], dtype=float)
    r = blkbt(R, n_dims=1)
    assert r.extra["n_respondents"] == 2
