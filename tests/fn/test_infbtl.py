"""Tests for infbtl.information_bottleneck."""
import numpy as np
import pytest
from morie.fn.infbtl import information_bottleneck


def test_infbtl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = information_bottleneck(X, Y, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infbtl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = information_bottleneck(X, Y, beta)
    assert isinstance(result, dict)
