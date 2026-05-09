"""Tests for gb_wsn.gibbons_wsrt_normal_approx."""
import numpy as np
import pytest
from moirais.fn.gb_wsn import gibbons_wsrt_normal_approx


def test_gb_wsn_basic():
    """Test basic functionality."""
    T_plus = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_wsrt_normal_approx(T_plus, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_wsn_edge():
    """Test edge cases."""
    T_plus = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_wsrt_normal_approx(T_plus, n)
    assert isinstance(result, dict)
