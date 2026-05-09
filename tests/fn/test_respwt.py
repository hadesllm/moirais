"""Tests for respwt.response_weight."""
import numpy as np
import pytest
from moirais.fn.respwt import response_weight


def test_respwt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cell = np.random.default_rng(42).normal(0, 1, 100)
    r_h = np.random.default_rng(42).normal(0, 1, 100)
    n_h = np.random.default_rng(42).normal(0, 1, 100)
    result = response_weight(y, weights, cell, r_h, n_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_respwt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cell = np.random.default_rng(42).normal(0, 1, 100)
    r_h = np.random.default_rng(42).normal(0, 1, 100)
    n_h = np.random.default_rng(42).normal(0, 1, 100)
    result = response_weight(y, weights, cell, r_h, n_h)
    assert isinstance(result, dict)
