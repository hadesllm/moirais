"""Tests for gb233.gibbons_edf_asymp_normal."""
import numpy as np
import pytest
from moirais.fn.gb233 import gibbons_edf_asymp_normal


def test_gb233_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_asymp_normal(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb233_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_asymp_normal(x, n)
    assert isinstance(result, dict)
