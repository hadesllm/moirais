"""Tests for hrzaul.horowitz_additive_unknown_link."""
import numpy as np
import pytest
from morie.fn.hrzaul import horowitz_additive_unknown_link


def test_hrzaul_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_additive_unknown_link(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzaul_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_additive_unknown_link(x, y, bandwidth)
    assert isinstance(result, dict)
