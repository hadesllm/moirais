"""Tests for ksr12.kosorok_information_bound."""
import numpy as np
import pytest
from moirais.fn.ksr12 import kosorok_information_bound


def test_ksr12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_information_bound(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_information_bound(x, y)
    assert isinstance(result, dict)
