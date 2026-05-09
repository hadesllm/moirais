"""Tests for mareml.ma_random_reml."""
import numpy as np
import pytest
from moirais.fn.mareml import ma_random_reml


def test_mareml_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_random_reml(yi, vi, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mareml_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_random_reml(yi, vi, max_iter)
    assert isinstance(result, dict)
