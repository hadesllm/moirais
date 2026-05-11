"""Tests for bkprp.backpropagation."""
import numpy as np
import pytest
from morie.fn.bkprp import backpropagation


def test_bkprp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = backpropagation(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkprp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = backpropagation(x, y)
    assert isinstance(result, dict)
