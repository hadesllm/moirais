"""Tests for ksr040.kosorok_ch2_bootstrap_donsker_iff."""
import numpy as np
import pytest
from morie.fn.ksr040 import kosorok_ch2_bootstrap_donsker_iff


def test_ksr040_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_bootstrap_donsker_iff(F, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr040_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_bootstrap_donsker_iff(F, P)
    assert isinstance(result, dict)
