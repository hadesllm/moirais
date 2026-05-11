"""Tests for asymptotic z-test."""
from morie.fn.sgzts import sgzts


def test_sgzts_smoke():
    r = sgzts(2.5, 0.0, 1.0)
    assert r.name == "asymptotic_z_test"
    assert abs(r.extra["z_statistic"] - 2.5) < 1e-10
    assert r.extra["p_value"] < 0.05


def test_sgzts_not_significant():
    r = sgzts(0.5, 0.0, 1.0)
    assert not r.extra["significant"]
