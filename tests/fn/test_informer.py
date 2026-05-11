"""Tests for informer.informer_long_horizon."""
import numpy as np
import pytest
from morie.fn.informer import informer_long_horizon


def test_informer_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = informer_long_horizon(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_informer_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = informer_long_horizon(y, horizon)
    assert isinstance(result, dict)
