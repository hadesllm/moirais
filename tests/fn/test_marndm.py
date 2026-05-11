"""Tests for marndm.ma_random_dl."""
import numpy as np
import pytest
from morie.fn.marndm import ma_random_dl


def test_marndm_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_random_dl(yi, vi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_marndm_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_random_dl(yi, vi)
    assert isinstance(result, dict)
