"""Tests for gb_frs.gibbons_friedman_chi2_approp."""
import numpy as np
import pytest
from moirais.fn.gb_frs import gibbons_friedman_chi2_approp


def test_gb_frs_basic():
    """Test basic functionality."""
    chi_r2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman_chi2_approp(chi_r2, k, b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_frs_edge():
    """Test edge cases."""
    chi_r2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman_chi2_approp(chi_r2, k, b)
    assert isinstance(result, dict)
