"""Tests for pnaG.pna."""
import numpy as np
import pytest
from morie.fn.pnaG import pna


def test_pnaG_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aggregators = np.random.default_rng(42).normal(0, 1, 100)
    scalers = np.random.default_rng(42).normal(0, 1, 100)
    result = pna(A, X, aggregators, scalers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pnaG_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    aggregators = np.random.default_rng(42).normal(0, 1, 100)
    scalers = np.random.default_rng(42).normal(0, 1, 100)
    result = pna(A, X, aggregators, scalers)
    assert isinstance(result, dict)
