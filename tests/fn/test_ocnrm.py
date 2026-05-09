"""Tests for moirais.fn.ocnrm -- OC normal vector."""
import numpy as np
from moirais.fn.ocnrm import oc_normal_vector, ocnrm


def test_alias():
    assert ocnrm is oc_normal_vector


def test_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 2))
    votes_j = np.array([1]*10 + [0]*10, dtype=float)
    r = oc_normal_vector(X, votes_j)
    assert r.name == "oc_normal_vector"
    assert "normal" in r.extra
    assert r.extra["n_yea"] == 10
    assert r.extra["n_nay"] == 10
