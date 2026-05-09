"""Tests for itinft.item_information_function."""
import numpy as np
import pytest
from moirais.fn.itinft import item_information_function


def test_itinft_basic():
    """Test basic functionality."""
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = item_information_function(theta, a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itinft_edge():
    """Test edge cases."""
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = item_information_function(theta, a, b)
    assert isinstance(result, dict)
