"""Tests for kfd — Katz fractal dimension."""
import numpy as np
from morie.fn.kfd import katz_fd
from morie.fn._containers import DescriptiveResult


def test_kfd_basic(rng):
    x = np.cumsum(rng.standard_normal(500))
    result = katz_fd(x)
    assert isinstance(result, DescriptiveResult)
    assert result.value > 1.0


def test_kfd_sine_finite():
    sine = np.sin(np.linspace(0, 10 * np.pi, 1000))
    result = katz_fd(sine)
    assert 1.0 < result.value < 3.0
