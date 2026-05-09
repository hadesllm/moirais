"""Tests for moirais.fn.bamse -- posterior standard errors."""
import numpy as np
from moirais.fn.bamse import bayesian_se_from_posterior, bamse


def test_alias():
    assert bamse is bayesian_se_from_posterior


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((200, 2))
    r = bayesian_se_from_posterior(chain)
    assert r.name == "bayesian_se_from_posterior"
    assert len(r.extra["ses"]) == 2
    assert all(s > 0 for s in r.extra["ses"])


def test_1d():
    chain = np.array([1.0, 2.0, 3.0, 4.0])
    r = bayesian_se_from_posterior(chain)
    assert r.value > 0
