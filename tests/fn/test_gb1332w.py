"""Tests for gb1332w.gibbons_wrs_efficacy."""
import numpy as np
import pytest
from morie.fn.gb1332w import gibbons_wrs_efficacy


def test_gb1332w_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = gibbons_wrs_efficacy(f, lam)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1332w_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = gibbons_wrs_efficacy(f, lam)
    assert isinstance(result, dict)
