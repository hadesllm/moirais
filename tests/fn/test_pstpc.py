"""Tests for moirais.fn.pstpc -- posterior predictive check."""
import numpy as np
from moirais.fn.pstpc import posterior_predictive_check, pstpc


def test_alias():
    assert pstpc is posterior_predictive_check


def test_smoke():
    chain = np.random.default_rng(42).standard_normal(200)
    data = np.random.default_rng(42).standard_normal(50)
    r = posterior_predictive_check(chain, data)
    assert r.name == "posterior_predictive_check"
    assert 0.0 <= r.value <= 1.0
    assert "ppp_value" in r.extra
