"""Tests for grohe.geron_one_hot_encoding."""
import numpy as np
import pytest
from morie.fn.grohe import geron_one_hot_encoding


def test_grohe_basic():
    """Test basic functionality."""
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_hot_encoding(categories)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grohe_edge():
    """Test edge cases."""
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_hot_encoding(categories)
    assert isinstance(result, dict)
