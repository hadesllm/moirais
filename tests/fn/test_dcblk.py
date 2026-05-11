"""Tests for dcblk -- DC blocker."""
import numpy as np
from morie.fn.dcblk import dcblk
from morie.fn._containers import SignalResult


def test_dcblk_mean_removal():
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    result = dcblk(x)
    assert isinstance(result, SignalResult)
    assert abs(np.mean(result.filtered)) < 1e-10


def test_dcblk_iir_method():
    x = np.ones(100) * 10.0
    result = dcblk(x, method="iir")
    assert abs(result.filtered[-1]) < 1.0


def test_dcblk_preserves_ac():
    rng = np.random.default_rng(42)
    t = np.arange(0, 1.0, 0.001)
    x = np.sin(2 * np.pi * 50 * t) + 5.0
    result = dcblk(x)
    assert np.std(result.filtered) > 0.5
    assert abs(np.mean(result.filtered)) < 1e-10
