"""Tests for mapaule.ma_paule_mandel."""
import numpy as np
import pytest
from morie.fn.mapaule import ma_paule_mandel


def test_mapaule_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_paule_mandel(yi, vi, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mapaule_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_paule_mandel(yi, vi, max_iter)
    assert isinstance(result, dict)
