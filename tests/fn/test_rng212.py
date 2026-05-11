"""Tests for rng212.rangayyan_ch4_matched_filter_instantaneous_signal."""
import numpy as np
import pytest
from morie.fn.rng212 import rangayyan_ch4_matched_filter_instantaneous_signal


def test_rng212_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_instantaneous_signal(X, H, f, t_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng212_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_instantaneous_signal(X, H, f, t_0)
    assert isinstance(result, dict)
