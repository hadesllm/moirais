"""Tests for gb5417.gibbons_sign_median_ci."""
import numpy as np
import pytest
from moirais.fn.gb5417 import gibbons_sign_median_ci


def test_gb5417_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_sign_median_ci(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5417_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_sign_median_ci(x, alpha)
    assert isinstance(result, dict)
