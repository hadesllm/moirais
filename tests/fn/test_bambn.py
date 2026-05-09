"""Tests for moirais.fn.bambn -- burn-in trimming."""
import numpy as np
from moirais.fn.bambn import bayesian_burnin_trim, bambn


def test_alias():
    assert bambn is bayesian_burnin_trim


def test_smoke():
    chain = np.arange(200, dtype=float)
    r = bayesian_burnin_trim(chain, burnin=50)
    assert r.name == "bayesian_burnin_trim"
    assert r.extra["trimmed_length"] == 150
    assert r.extra["original_length"] == 200


def test_default_burnin():
    chain = np.arange(300, dtype=float)
    r = bayesian_burnin_trim(chain)
    assert r.extra["burnin"] == 100
    assert r.extra["trimmed_length"] == 200
