"""Tests for pfd — Petrosian fractal dimension."""
import numpy as np
from moirais.fn.pfd import petrosian_fd
from moirais.fn._containers import DescriptiveResult


def test_pfd_basic(rng):
    x = rng.standard_normal(500)
    result = petrosian_fd(x)
    assert isinstance(result, DescriptiveResult)
    assert 1.0 <= result.value <= 1.1


def test_pfd_noisy_higher_than_smooth(rng):
    noisy = rng.standard_normal(1000)
    smooth = np.sin(np.linspace(0, 2 * np.pi, 1000))
    d_noisy = petrosian_fd(noisy).value
    d_smooth = petrosian_fd(smooth).value
    assert d_noisy > d_smooth
