"""Tests for hetero.htmt_ratio."""
import numpy as np
import pytest
from moirais.fn.hetero import htmt_ratio


def test_hetero_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    construct_assignment = np.random.default_rng(42).normal(0, 1, 100)
    result = htmt_ratio(X, construct_assignment)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hetero_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    construct_assignment = np.random.default_rng(42).normal(0, 1, 100)
    result = htmt_ratio(X, construct_assignment)
    assert isinstance(result, dict)
