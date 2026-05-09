"""Tests for moirais.fn.chnvg -- chain convergence test."""
import numpy as np
from moirais.fn.chnvg import chain_convergence_test, chnvg


def test_alias():
    assert chnvg is chain_convergence_test


def test_converged():
    rng = np.random.default_rng(42)
    chains = [rng.standard_normal(200) for _ in range(4)]
    r = chain_convergence_test(chains)
    assert r.name == "chain_convergence_test"
    assert r.extra["converged"] is True
    assert r.value < 1.1


def test_diverged():
    chains = [np.full(100, 0.0), np.full(100, 10.0)]
    r = chain_convergence_test(chains)
    assert r.extra["converged"] is False
