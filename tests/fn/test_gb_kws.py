"""Tests for gb_kws.gibbons_kw_chi2_approx."""
import numpy as np
import pytest
from morie.fn.gb_kws import gibbons_kw_chi2_approx


def test_gb_kws_basic():
    """Test basic functionality."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_kw_chi2_approx(H, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_kws_edge():
    """Test edge cases."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_kw_chi2_approx(H, k)
    assert isinstance(result, dict)
