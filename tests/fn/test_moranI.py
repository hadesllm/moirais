"""Tests for moranI.morans_i_asymptotic_test."""
import numpy as np
import pytest
from moirais.fn.moranI import morans_i_asymptotic_test


def test_moranI_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i_asymptotic_test(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_moranI_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i_asymptotic_test(x, W)
    assert isinstance(result, dict)
