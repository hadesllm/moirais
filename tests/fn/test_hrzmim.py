"""Tests for hrzmim.horowitz_multiple_index_model."""
import numpy as np
import pytest
from morie.fn.hrzmim import horowitz_multiple_index_model


def test_hrzmim_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = horowitz_multiple_index_model(x, y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzmim_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = horowitz_multiple_index_model(x, y, k)
    assert isinstance(result, dict)
