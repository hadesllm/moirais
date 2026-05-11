"""Tests for kde_smooth."""
import numpy as np
from morie.fn.kdesm import kde_smooth, kdesm


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r = kde_smooth(x)
    assert r.value > 0
    assert len(r.extra["grid"]) == 256


def test_alias():
    assert kdesm is kde_smooth


def test_custom_bandwidth():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = kde_smooth(x, bandwidth=0.5)
    assert abs(r.extra["bandwidth"] - 0.5) < 1e-10
