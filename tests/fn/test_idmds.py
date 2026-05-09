"""Tests for INDSCAL."""
import numpy as np
from moirais.fn.idmds import idmds


def test_idmds_smoke():
    rng = np.random.default_rng(42)
    n = 5
    dists = []
    for _ in range(3):
        D = rng.random((n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        dists.append(D)
    r = idmds(dists, n_dims=2, max_iter=20)
    assert r.name == "indscal_mds"
    assert "group_config" in r.extra
    assert "weights" in r.extra
    assert r.extra["weights"].shape == (3, 2)


def test_cheatsheet():
    from moirais.fn.idmds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
