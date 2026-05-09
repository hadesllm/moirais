"""Tests for fzmiq.fauzi_moment_ineq_ustat."""
import numpy as np
import pytest
from moirais.fn.fzmiq import fauzi_moment_ineq_ustat


def test_fzmiq_basic():
    """Test basic functionality."""
    u_stat_func = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_moment_ineq_ustat(u_stat_func, q)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzmiq_edge():
    """Test edge cases."""
    u_stat_func = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_moment_ineq_ustat(u_stat_func, q)
    assert isinstance(result, dict)
