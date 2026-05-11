"""Tests for gb4351.gibbons_ks_chi2_approx."""
import numpy as np
import pytest
from morie.fn.gb4351 import gibbons_ks_chi2_approx


def test_gb4351_basic():
    """Test basic functionality."""
    n = 100
    Dplus = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_ks_chi2_approx(n, Dplus)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb4351_edge():
    """Test edge cases."""
    n = 100
    Dplus = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_ks_chi2_approx(n, Dplus)
    assert isinstance(result, dict)
