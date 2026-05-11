"""Tests for grgruc.geron_gru_cell."""
import numpy as np
import pytest
from morie.fn.grgruc import geron_gru_cell


def test_grgruc_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wz = np.random.default_rng(42).normal(0, 1, 100)
    Wr = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgruc_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wz = np.random.default_rng(42).normal(0, 1, 100)
    Wr = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)
