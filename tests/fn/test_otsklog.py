"""Tests for otsklog.ot_sinkhorn_log."""
import numpy as np
import pytest
from morie.fn.otsklog import ot_sinkhorn_log


def test_otsklog_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_log(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsklog_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_log(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
