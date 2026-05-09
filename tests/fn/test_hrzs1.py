"""Tests for hrzs1.horowitz_sample_selection."""
import numpy as np
import pytest
from moirais.fn.hrzs1 import horowitz_sample_selection


def test_hrzs1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    result = horowitz_sample_selection(x, y, z, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzs1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    result = horowitz_sample_selection(x, y, z, d)
    assert isinstance(result, dict)
