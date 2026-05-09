"""Tests for mrmr_score."""
import numpy as np
from moirais.fn.mirmf import mrmr_score, mirmf


def test_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = X[:, 0] + rng.normal(0, 0.1, 100)
    r = mrmr_score(X, y, n_features=3)
    assert len(r.extra["selected_indices"]) == 3
    assert 0 in r.extra["selected_indices"]


def test_alias():
    assert mirmf is mrmr_score
