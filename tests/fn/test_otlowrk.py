"""Tests for otlowrk.ot_low_rank_sinkhorn."""
import numpy as np
import pytest
from moirais.fn.otlowrk import ot_low_rank_sinkhorn


def test_otlowrk_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    rank = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_low_rank_sinkhorn(a, b, C, rank, epsilon, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otlowrk_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    rank = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_low_rank_sinkhorn(a, b, C, rank, epsilon, max_iter)
    assert isinstance(result, dict)
