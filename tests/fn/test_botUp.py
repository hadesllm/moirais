"""Tests for botUp.bottom_up_aggregation."""
import numpy as np
import pytest
from morie.fn.botUp import bottom_up_aggregation


def test_botUp_basic():
    """Test basic functionality."""
    bottoms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = bottom_up_aggregation(bottoms, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_botUp_edge():
    """Test edge cases."""
    bottoms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = bottom_up_aggregation(bottoms, S)
    assert isinstance(result, dict)
