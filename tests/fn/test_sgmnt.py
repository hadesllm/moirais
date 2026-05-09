"""Tests for Mantel test."""
import numpy as np
from moirais.fn.sgmnt import sgmnt


def test_sgmnt_smoke():
    rng = np.random.default_rng(42)
    n = 20
    Ds = rng.uniform(0, 10, (n, n))
    Ds = (Ds + Ds.T) / 2
    np.fill_diagonal(Ds, 0)
    Da = rng.uniform(0, 5, (n, n))
    Da = (Da + Da.T) / 2
    np.fill_diagonal(Da, 0)
    r = sgmnt(Ds, Da, n_perm=99)
    assert r.name == "mantel_test"
    assert "p_value" in r.extra
    assert 0 <= r.extra["p_value"] <= 1


def test_cheatsheet():
    from moirais.fn.sgmnt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
