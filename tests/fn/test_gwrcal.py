"""Tests for gwrcal.gwr_bandwidth_select."""
import numpy as np
import pytest
from moirais.fn.gwrcal import gwr_bandwidth_select


def test_gwrcal_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gwr_bandwidth_select(y, X, coords, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwrcal_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gwr_bandwidth_select(y, X, coords, kernel)
    assert isinstance(result, dict)
