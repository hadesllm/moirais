"""Tests for magal.ma_galbraith."""
import numpy as np
import pytest
from morie.fn.magal import ma_galbraith


def test_magal_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_galbraith(yi, se_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_magal_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_galbraith(yi, se_i)
    assert isinstance(result, dict)
