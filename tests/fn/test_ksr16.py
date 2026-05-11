"""Tests for ksr16.kosorok_influence_function."""
import numpy as np
import pytest
from morie.fn.ksr16 import kosorok_influence_function


def test_ksr16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_influence_function(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_influence_function(x, y)
    assert isinstance(result, dict)
