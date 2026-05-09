"""Tests for maloo.ma_leave_one_out."""
import numpy as np
import pytest
from moirais.fn.maloo import ma_leave_one_out


def test_maloo_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = ma_leave_one_out(yi, vi, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_maloo_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = ma_leave_one_out(yi, vi, method)
    assert isinstance(result, dict)
