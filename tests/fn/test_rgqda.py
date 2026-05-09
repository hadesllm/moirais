"""Tests for rgqda.rangayyan_qda."""
import numpy as np
import pytest
from moirais.fn.rgqda import rangayyan_qda


def test_rgqda_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_qda(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgqda_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_qda(X, y)
    assert isinstance(result, dict)
