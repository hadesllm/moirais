"""Tests for PQL spatial GLMM."""
import numpy as np
from moirais.fn.sgpql import sgpql


def test_sgpql_smoke():
    rng = np.random.default_rng(20)
    n = 15
    coords = rng.uniform(0, 5, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    Z = rng.poisson(3, n).astype(float)
    r = sgpql(Z, X, coords, family="poisson")
    assert r.name == "pql_spatial_glmm"
    assert "beta" in r.extra
    assert "random_effects" in r.extra


def test_cheatsheet():
    from moirais.fn.sgpql import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
