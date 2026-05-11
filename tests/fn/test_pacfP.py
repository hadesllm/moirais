"""Tests for pacfP.partial_autocorrelation."""
import numpy as np
import pytest
from morie.fn.pacfP import partial_autocorrelation


def test_pacfP_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag_max = 100
    result = partial_autocorrelation(y, lag_max)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_pacfP_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag_max = 100
    result = partial_autocorrelation(y, lag_max)
    assert isinstance(result, dict)
