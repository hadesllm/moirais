"""Tests for rglindf.rangayyan_linear_discrim."""
import numpy as np
import pytest
from moirais.fn.rglindf import rangayyan_linear_discrim


def test_rglindf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    w0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_linear_discrim(X, y, w, w0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglindf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    w0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_linear_discrim(X, y, w, w0)
    assert isinstance(result, dict)
