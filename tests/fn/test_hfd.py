"""Tests for hfd — Higuchi fractal dimension."""
import numpy as np
from morie.fn.hfd import higuchi_fd
from morie.fn._containers import DescriptiveResult


def test_hfd_basic(rng):
    x = np.cumsum(rng.standard_normal(1000))
    result = higuchi_fd(x)
    assert isinstance(result, DescriptiveResult)
    assert result.value is not None


def test_hfd_brownian_motion(rng):
    x = np.cumsum(rng.standard_normal(2000))
    result = higuchi_fd(x, kmax=20)
    assert 1.3 < result.value < 1.7


def test_hfd_sine_low_dimension():
    t = np.linspace(0, 10 * np.pi, 2000)
    x = np.sin(t)
    result = higuchi_fd(x, kmax=20)
    assert result.value < 1.3
