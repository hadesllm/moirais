"""Tests for raklng.raking_ratio."""
import numpy as np
import pytest
from morie.fn.raklng import raking_ratio


def test_raklng_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    margins = np.random.default_rng(42).normal(0, 1, 100)
    result = raking_ratio(y, weights, margins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_raklng_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    margins = np.random.default_rng(42).normal(0, 1, 100)
    result = raking_ratio(y, weights, margins)
    assert isinstance(result, dict)
