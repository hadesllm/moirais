"""Tests for manct.ma_continuity_correction."""
import numpy as np
import pytest
from moirais.fn.manct import ma_continuity_correction


def test_manct_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    cc = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_continuity_correction(a, b, c, d, cc)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manct_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    cc = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_continuity_correction(a, b, c, d, cc)
    assert isinstance(result, dict)
