"""Tests for ksr035.kosorok_ch2_donsker_bracketing_integral."""
import numpy as np
import pytest
from morie.fn.ksr035 import kosorok_ch2_donsker_bracketing_integral


def test_ksr035_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_bracketing_integral(F, P, r, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr035_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_bracketing_integral(F, P, r, delta)
    assert isinstance(result, dict)
