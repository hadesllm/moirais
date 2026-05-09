"""Tests for hmmcel.geron_memory_cell."""
import numpy as np
import pytest
from moirais.fn.hmmcel import geron_memory_cell


def test_hmmcel_basic():
    """Test basic functionality."""
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_memory_cell(c_prev, x_t, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmcel_edge():
    """Test edge cases."""
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_memory_cell(c_prev, x_t, f)
    assert isinstance(result, dict)
