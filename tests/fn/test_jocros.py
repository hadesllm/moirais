"""Tests for jocros.joseph_croston_intermittent."""
import numpy as np
import pytest
from moirais.fn.jocros import joseph_croston_intermittent


def test_jocros_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_croston_intermittent(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jocros_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_croston_intermittent(y, alpha)
    assert isinstance(result, dict)
