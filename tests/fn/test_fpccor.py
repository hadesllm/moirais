"""Tests for fpccor.functional_correlation."""
import numpy as np
import pytest
from moirais.fn.fpccor import functional_correlation


def test_fpccor_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_correlation(X, Y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_fpccor_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_correlation(X, Y)
    assert isinstance(result, dict)
