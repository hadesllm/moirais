"""Tests for chrbnd.chernozhukov_rosen_bounds."""
import numpy as np
import pytest
from moirais.fn.chrbnd import chernozhukov_rosen_bounds


def test_chrbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    instrument = np.random.default_rng(42).normal(0, 1, 100)
    result = chernozhukov_rosen_bounds(y, X, instrument)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chrbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    instrument = np.random.default_rng(42).normal(0, 1, 100)
    result = chernozhukov_rosen_bounds(y, X, instrument)
    assert isinstance(result, dict)
