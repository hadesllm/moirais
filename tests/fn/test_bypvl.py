"""Tests for morie.fn.bypvl -- Bayesian p-value."""
import numpy as np
from morie.fn.bypvl import bayesian_p_value, bypvl


def test_alias():
    assert bypvl is bayesian_p_value


def test_smoke():
    chain = np.random.default_rng(42).standard_normal(500)
    r = bayesian_p_value(chain, test_stat=0.0)
    assert r.name == "bayesian_p_value"
    assert 0.0 <= r.value <= 1.0


def test_extreme():
    chain = np.ones(100)
    r = bayesian_p_value(chain, test_stat=2.0)
    assert r.value == 0.0
