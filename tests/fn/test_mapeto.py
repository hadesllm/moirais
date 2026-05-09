"""Tests for mapeto.ma_peto_or."""
import numpy as np
import pytest
from moirais.fn.mapeto import ma_peto_or


def test_mapeto_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_peto_or(a, b, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mapeto_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = ma_peto_or(a, b, c, d)
    assert isinstance(result, dict)
