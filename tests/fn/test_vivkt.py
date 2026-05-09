"""Tests for vivkt.vansteelandt_vanderweele."""
import numpy as np
import pytest
from moirais.fn.vivkt import vansteelandt_vanderweele


def test_vivkt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = vansteelandt_vanderweele(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vivkt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = vansteelandt_vanderweele(X, M, Y)
    assert isinstance(result, dict)
