"""Tests for hmonl.geron_online_learning."""
import numpy as np
import pytest
from morie.fn.hmonl import geron_online_learning


def test_hmonl_basic():
    """Test basic functionality."""
    X_stream = np.random.default_rng(42).normal(0, 1, 100)
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_online_learning(X_stream, y_stream, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmonl_edge():
    """Test edge cases."""
    X_stream = np.random.default_rng(42).normal(0, 1, 100)
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_online_learning(X_stream, y_stream, eta)
    assert isinstance(result, dict)
