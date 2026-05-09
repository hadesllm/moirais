"""Tests for moirais.fn.effsz -- effective sample size."""
import numpy as np
from moirais.fn.effsz import effective_sample_size, effsz


def test_alias():
    assert effsz is effective_sample_size


def test_iid():
    chain = np.random.default_rng(42).standard_normal(500)
    r = effective_sample_size(chain)
    assert r.name == "effective_sample_size"
    assert r.value > 100


def test_correlated():
    x = np.cumsum(np.random.default_rng(42).standard_normal(500))
    r = effective_sample_size(x)
    assert r.value < 500
