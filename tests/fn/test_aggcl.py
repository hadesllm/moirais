"""Tests for aggcl (agglomerative clustering)."""
import numpy as np
from moirais.fn.aggcl import agglomerative


def test_agglomerative_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))
    r = agglomerative(X, n_clusters=3)
    labels = r.extra["labels"]
    assert len(labels) == 20
    assert len(set(labels)) <= 3


def test_cheatsheet():
    from moirais.fn.aggcl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
