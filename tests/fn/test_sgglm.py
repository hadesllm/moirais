"""Tests for spatial GLM Poisson."""
import numpy as np
from moirais.fn.sgglm import sgglm


def test_sgglm_smoke():
    rng = np.random.default_rng(18)
    n = 20
    coords = rng.uniform(0, 10, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    counts = rng.poisson(5, n).astype(float)
    r = sgglm(counts, X, coords)
    assert r.name == "spatial_glm_poisson"
    assert "beta" in r.extra
    assert "deviance_residuals" in r.extra


def test_cheatsheet():
    from moirais.fn.sgglm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
