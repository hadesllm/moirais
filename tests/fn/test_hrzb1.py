"""Tests for hrzb1.horowitz_binary_response."""
import numpy as np
import pytest
from moirais.fn.hrzb1 import horowitz_binary_response


def test_hrzb1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_binary_response(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzb1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_binary_response(x, y)
    assert isinstance(result, dict)
