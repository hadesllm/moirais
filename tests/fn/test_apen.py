"""Tests for apen — Approximate entropy."""
import numpy as np
from morie.fn.apen import approx_entropy
from morie.fn._containers import DescriptiveResult


def test_apen_basic(rng):
    x = rng.standard_normal(200)
    result = approx_entropy(x, m=2, r=0.2)
    assert isinstance(result, DescriptiveResult)
    assert result.value >= 0


def test_apen_periodic_low():
    x = np.tile([1.0, 2.0, 3.0, 2.0], 50)
    result = approx_entropy(x, m=2, r=0.3)
    assert result.value < 1.0


def test_apen_random_higher(rng):
    periodic = np.tile([1.0, 2.0, 3.0, 2.0], 50)
    random = rng.standard_normal(200)
    ap_periodic = approx_entropy(periodic, m=2, r=0.3).value
    ap_random = approx_entropy(random, m=2, r=0.3).value
    assert ap_random > ap_periodic
