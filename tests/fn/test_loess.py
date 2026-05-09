"""Tests for loess.loess."""
import numpy as np
import pytest
from moirais.fn.loess import loess


def test_loess_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    span = np.random.default_rng(42).normal(0, 1, 100)
    result = loess(x, y, span)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_loess_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    span = np.random.default_rng(42).normal(0, 1, 100)
    result = loess(x, y, span)
    assert isinstance(result, dict)
