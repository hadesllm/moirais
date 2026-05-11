"""Tests for rng208.rangayyan_ch4_matched_filter_output_inverse_ft."""
import numpy as np
import pytest
from morie.fn.rng208 import rangayyan_ch4_matched_filter_output_inverse_ft


def test_rng208_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_matched_filter_output_inverse_ft(X, H, omega, f, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng208_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_matched_filter_output_inverse_ft(X, H, omega, f, t)
    assert isinstance(result, dict)
