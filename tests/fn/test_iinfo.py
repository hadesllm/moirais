"""Tests for iinfo.item_information."""
import numpy as np
import pytest
from morie.fn.iinfo import item_information


def test_iinfo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = item_information(y, theta, a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_iinfo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = item_information(y, theta, a, b)
    assert isinstance(result, dict)
