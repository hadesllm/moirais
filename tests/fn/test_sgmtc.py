"""Tests for Matern cluster process."""
from morie.fn.sgmtc import sgmtc


def test_sgmtc_smoke():
    r = sgmtc(5.0, 10, 1.0, (0, 10, 0, 10), seed=42)
    assert r.name == "matern_cluster_process"
    assert "points" in r.extra
    assert r.extra["n_points"] > 0
    assert r.extra["n_parents"] > 0


def test_cheatsheet():
    from morie.fn.sgmtc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
