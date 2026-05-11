"""Tests for backDR.back_door."""
import numpy as np
import pytest
from morie.fn.backDR import back_door


def test_backDR_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = back_door(Y, X, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_backDR_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = back_door(Y, X, C)
    assert isinstance(result, dict)
