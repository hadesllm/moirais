"""Tests for vrtest.variance_component_lr_boundary."""
import numpy as np
import pytest
from moirais.fn.vrtest import variance_component_lr_boundary


def test_vrtest_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_component_lr_boundary(y, X, Z, cluster)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_vrtest_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_component_lr_boundary(y, X, Z, cluster)
    assert isinstance(result, dict)
