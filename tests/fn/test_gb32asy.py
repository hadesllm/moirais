"""Tests for gb32asy.gibbons_runs_asymp_normal."""
import numpy as np
import pytest
from moirais.fn.gb32asy import gibbons_runs_asymp_normal


def test_gb32asy_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_asymp_normal(R, n, n1, n2)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb32asy_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_asymp_normal(R, n, n1, n2)
    assert isinstance(result, dict)
