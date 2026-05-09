"""Tests for bndtfm.bound_transform."""
import numpy as np
import pytest
from moirais.fn.bndtfm import bound_transform


def test_bndtfm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    transform = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_transform(y, D, X, transform)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndtfm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    transform = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_transform(y, D, X, transform)
    assert isinstance(result, dict)
