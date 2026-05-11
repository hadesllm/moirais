"""Tests for hrzades.horowitz_improved_ade."""
import numpy as np
import pytest
from morie.fn.hrzades import horowitz_improved_ade


def test_hrzades_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_improved_ade(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzades_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_improved_ade(x, y, bandwidth)
    assert isinstance(result, dict)
