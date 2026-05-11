"""Tests for morie.fn.smcnv -- SMACOF convergence."""

from morie.fn.smcnv import smacof_convergence, smcnv


def test_smcnv_converged():
    r = smcnv(0.01, 0.01, tol=1e-6)
    assert r.name == "smacof_convergence"
    assert r.value is True


def test_smcnv_not_converged():
    r = smcnv(0.1, 0.01, tol=1e-6)
    assert r.value is False


def test_smcnv_alias():
    assert smcnv is smacof_convergence
