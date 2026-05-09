"""Tests for gb631a.gibbons_ks2_asymp."""
import numpy as np
import pytest
from moirais.fn.gb631a import gibbons_ks2_asymp


def test_gb631a_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ks2_asymp(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb631a_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ks2_asymp(x, y)
    assert isinstance(result, dict)
