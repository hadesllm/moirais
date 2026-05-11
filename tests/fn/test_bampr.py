"""Tests for morie.fn.bampr -- posterior summary."""
import numpy as np
from morie.fn.bampr import bayesian_am_posterior_summary, bampr


def test_alias():
    assert bampr is bayesian_am_posterior_summary


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((200, 3))
    r = bayesian_am_posterior_summary(chain)
    assert r.name == "bayesian_am_posterior_summary"
    assert len(r.extra["means"]) == 3
    assert len(r.extra["sds"]) == 3
    assert len(r.extra["ci_lo"]) == 3


def test_1d_chain():
    chain = np.random.default_rng(42).standard_normal(100)
    r = bayesian_am_posterior_summary(chain)
    assert r.extra["n_params"] == 1
