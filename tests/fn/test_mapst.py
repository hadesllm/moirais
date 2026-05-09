"""Tests for map_estimate."""
import numpy as np
import pytest
from moirais.fn.mapst import map_estimate, mapst


def test_shrinkage():
    x = np.array([10.0, 10.0, 10.0])
    r = map_estimate(x, prior_mu=0.0, prior_sigma=1.0)
    assert r.estimate < 10.0
    assert r.estimate > 0.0


def test_alias():
    assert mapst is map_estimate


def test_bad_sigma():
    with pytest.raises(ValueError):
        map_estimate([1, 2], prior_sigma=0.0)


def test_large_n_approaches_mle():
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, 10000)
    r = map_estimate(x, prior_mu=0.0, prior_sigma=1.0)
    assert abs(r.estimate - 5.0) < 0.1
