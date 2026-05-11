"""Tests for pysrSR.pysr_regression."""
import numpy as np
import pytest
from morie.fn.pysrSR import pysr_regression


def test_pysrSR_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = pysr_regression(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pysrSR_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = pysr_regression(X, y)
    assert isinstance(result, dict)
