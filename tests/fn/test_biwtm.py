"""Test biwtm."""
import numpy as np
import pytest
from morie.fn.biwtm import biweight_midcorrelation


def test_biwtm_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = x + rng.standard_normal(50) * 0.3
    r = biweight_midcorrelation(x, y)
    assert -1.0 <= r.value <= 1.0
    assert r.name == "biwtm"


def test_biwtm_uncorrelated():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    r = biweight_midcorrelation(x, y)
    assert abs(r.value) < 0.3
