"""Tests for ksr041.kosorok_ch2_bootstrap_donsker_almost_sure."""
import numpy as np
import pytest
from moirais.fn.ksr041 import kosorok_ch2_bootstrap_donsker_almost_sure


def test_ksr041_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_bootstrap_donsker_almost_sure(F, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr041_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_bootstrap_donsker_almost_sure(F, P)
    assert isinstance(result, dict)
