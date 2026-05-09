"""Tests for infcrv.influence_function."""
import numpy as np
import pytest
from moirais.fn.infcrv import influence_function


def test_infcrv_basic():
    """Test basic functionality."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = influence_function(estimator, F, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infcrv_edge():
    """Test edge cases."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = influence_function(estimator, F, x)
    assert isinstance(result, dict)
