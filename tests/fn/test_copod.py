"""Tests for copod.copod."""
import numpy as np
import pytest
from moirais.fn.copod import copod


def test_copod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = copod(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = copod(X)
    assert isinstance(result, dict)
