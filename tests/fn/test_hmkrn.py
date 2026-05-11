"""Tests for hmkrn.geron_filter_kernel."""
import numpy as np
import pytest
from morie.fn.hmkrn import geron_filter_kernel


def test_hmkrn_basic():
    """Test basic functionality."""
    kh = np.random.default_rng(42).normal(0, 1, 100)
    kw = np.random.default_rng(42).normal(0, 1, 100)
    c_in = np.random.default_rng(42).normal(0, 1, 100)
    c_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_filter_kernel(kh, kw, c_in, c_out, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkrn_edge():
    """Test edge cases."""
    kh = np.random.default_rng(42).normal(0, 1, 100)
    kw = np.random.default_rng(42).normal(0, 1, 100)
    c_in = np.random.default_rng(42).normal(0, 1, 100)
    c_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_filter_kernel(kh, kw, c_in, c_out, seed)
    assert isinstance(result, dict)
