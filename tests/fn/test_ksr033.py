"""Tests for ksr033.kosorok_ch2_uniform_covering_number."""
import numpy as np
import pytest
from morie.fn.ksr033 import kosorok_ch2_uniform_covering_number


def test_ksr033_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = kosorok_ch2_uniform_covering_number(F, eps, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr033_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = kosorok_ch2_uniform_covering_number(F, eps, r)
    assert isinstance(result, dict)
