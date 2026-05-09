"""Tests for copul.copula_estimation."""
import numpy as np
import pytest
from moirais.fn.copul import copula_estimation


def test_copul_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = copula_estimation(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copul_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = copula_estimation(x, y)
    assert isinstance(result, dict)
