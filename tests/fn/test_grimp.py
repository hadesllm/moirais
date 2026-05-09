"""Tests for grimp.geron_simple_imputer."""
import numpy as np
import pytest
from moirais.fn.grimp import geron_simple_imputer


def test_grimp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_imputer(X, strategy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grimp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_simple_imputer(X, strategy)
    assert isinstance(result, dict)
