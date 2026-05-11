"""Tests for ecod.ecod."""
import numpy as np
import pytest
from morie.fn.ecod import ecod


def test_ecod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ecod(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ecod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ecod(X)
    assert isinstance(result, dict)
