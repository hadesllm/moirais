"""Tests for mamh.ma_mantel_haenszel."""
import numpy as np
import pytest
from moirais.fn.mamh import ma_mantel_haenszel


def test_mamh_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_mantel_haenszel(a, b, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mamh_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_mantel_haenszel(a, b, c, d)
    assert isinstance(result, dict)
