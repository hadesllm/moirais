"""Tests for morie.fn.bamth -- chain thinning."""
import numpy as np
from morie.fn.bamth import bayesian_thinning, bamth


def test_alias():
    assert bamth is bayesian_thinning


def test_smoke():
    chain = np.arange(100, dtype=float)
    r = bayesian_thinning(chain, thin=5)
    assert r.name == "bayesian_thinning"
    assert r.extra["thinned_length"] == 20
    assert r.extra["thin"] == 5


def test_thin_1():
    chain = np.arange(10, dtype=float)
    r = bayesian_thinning(chain, thin=1)
    assert r.extra["thinned_length"] == 10
