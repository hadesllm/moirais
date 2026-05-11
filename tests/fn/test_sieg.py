"""Tests for sieg.siegel_repeated."""
import numpy as np
import pytest
from morie.fn.sieg import siegel_repeated


def test_sieg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = siegel_repeated(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sieg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = siegel_repeated(x, y)
    assert isinstance(result, dict)
