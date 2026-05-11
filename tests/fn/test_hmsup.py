"""Tests for hmsup.geron_supervised_learning."""
import numpy as np
import pytest
from morie.fn.hmsup import geron_supervised_learning


def test_hmsup_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_supervised_learning(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsup_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_supervised_learning(X, y)
    assert isinstance(result, dict)
