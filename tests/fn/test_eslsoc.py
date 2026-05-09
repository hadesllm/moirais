"""Tests for eslsoc.esl_self_organize."""
import numpy as np
import pytest
from moirais.fn.eslsoc import esl_self_organize


def test_eslsoc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    grid = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_self_organize(X, grid, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsoc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    grid = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_self_organize(X, grid, eta)
    assert isinstance(result, dict)
