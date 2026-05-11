"""Tests for morie.fn.gelrb -- Gelman-Rubin R-hat."""
import numpy as np
from morie.fn.gelrb import gelman_rubin_rhat, gelrb


def test_alias():
    assert gelrb is gelman_rubin_rhat


def test_converged():
    rng = np.random.default_rng(42)
    chains = [rng.standard_normal(200) for _ in range(4)]
    r = gelman_rubin_rhat(chains)
    assert r.name == "gelman_rubin_rhat"
    assert r.value < 1.2


def test_diverged():
    chains = [np.full(100, 0.0), np.full(100, 10.0)]
    r = gelman_rubin_rhat(chains)
    assert r.value > 1.5
