"""Tests for moirais.fn.acfmc -- MCMC autocorrelation."""
import numpy as np
from moirais.fn.acfmc import autocorrelation_mcmc, acfmc


def test_alias():
    assert acfmc is autocorrelation_mcmc


def test_smoke():
    chain = np.random.default_rng(42).standard_normal(200)
    r = autocorrelation_mcmc(chain, max_lag=20)
    assert r.name == "autocorrelation_mcmc"
    assert len(r.extra["acf"]) == 21
    assert abs(r.extra["acf"][0] - 1.0) < 1e-10


def test_lag0_is_one():
    r = autocorrelation_mcmc(np.arange(50, dtype=float))
    assert abs(r.extra["acf"][0] - 1.0) < 1e-10
