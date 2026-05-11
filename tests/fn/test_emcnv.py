"""Tests for morie.fn.emcnv -- EM convergence check."""
from morie.fn.emcnv import em_convergence_check, emcnv


def test_alias():
    assert emcnv is em_convergence_check


def test_converged():
    r = em_convergence_check(-100.0, -100.0000001)
    assert r.name == "em_convergence_check"
    assert r.extra["converged"] is True


def test_not_converged():
    r = em_convergence_check(-100.0, -90.0)
    assert r.extra["converged"] is False
    assert r.extra["diff"] == 10.0
