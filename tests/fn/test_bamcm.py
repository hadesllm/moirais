"""Tests for morie.fn.bamcm -- Bayesian MCMC sampler."""
import numpy as np
from morie.fn.bamcm import bayesian_am_mcmc_sample, bamcm


def test_alias():
    assert bamcm is bayesian_am_mcmc_sample


def test_smoke():
    Z = np.random.default_rng(42).standard_normal((20, 2))
    r = bayesian_am_mcmc_sample(Z, n_samples=100)
    assert r.name == "bayesian_am_mcmc_sample"
    assert r.extra["chain"].shape == (100, 2)
    assert 0 < r.extra["acceptance_rate"] <= 1.0


def test_n_samples():
    Z = np.random.default_rng(42).standard_normal((10, 1))
    r = bayesian_am_mcmc_sample(Z, n_samples=50)
    assert r.extra["n_samples"] == 50
