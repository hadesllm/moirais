"""Tests for moirais.fn.mlscv -- MLSMU6 convergence check."""

from moirais.fn.mlscv import mlsmu6_convergence_check, mlscv


def test_mlscv_converged():
    r = mlscv(0.05, 0.05, tol=1e-4)
    assert r.name == "mlsmu6_convergence_check"
    assert r.value is True


def test_mlscv_not_converged():
    r = mlscv(0.1, 0.01, tol=1e-6)
    assert r.value is False


def test_mlscv_alias():
    assert mlscv is mlsmu6_convergence_check
