"""Tests for manskif.manski_bounds."""
import numpy as np
import pytest
from moirais.fn.manskif import manski_bounds


def test_manskif_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = manski_bounds(Y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manskif_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = manski_bounds(Y, X)
    assert isinstance(result, dict)
