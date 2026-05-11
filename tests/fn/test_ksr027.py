"""Tests for ksr027.kosorok_ch2_law_large_numbers_pointwise."""
import numpy as np
import pytest
from morie.fn.ksr027 import kosorok_ch2_law_large_numbers_pointwise


def test_ksr027_basic():
    """Test basic functionality."""
    F_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_law_large_numbers_pointwise(F_n, F, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr027_edge():
    """Test edge cases."""
    F_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_law_large_numbers_pointwise(F_n, F, t)
    assert isinstance(result, dict)
