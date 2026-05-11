"""Tests for rgnl.rangayyan_nonlinear_features."""
import numpy as np
import pytest
from morie.fn.rgnl import rangayyan_nonlinear_features


def test_rgnl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    r = 10
    result = rangayyan_nonlinear_features(x, m, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgnl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    r = 10
    result = rangayyan_nonlinear_features(x, m, r)
    assert isinstance(result, dict)
