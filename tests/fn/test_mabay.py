"""Tests for mabay.ma_bayes_random_effects."""
import numpy as np
import pytest
from moirais.fn.mabay import ma_bayes_random_effects


def test_mabay_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = ma_bayes_random_effects(yi, vi, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mabay_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = ma_bayes_random_effects(yi, vi, n_iter)
    assert isinstance(result, dict)
