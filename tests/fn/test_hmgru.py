"""Tests for hmgru.geron_gru."""
import numpy as np
import pytest
from moirais.fn.hmgru import geron_gru


def test_hmgru_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_gru(x_t, h_prev, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgru_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_gru(x_t, h_prev, weights)
    assert isinstance(result, dict)
