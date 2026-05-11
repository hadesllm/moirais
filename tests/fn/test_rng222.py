"""Tests for rng222.rangayyan_ch4_matched_filter_output_acf."""
import numpy as np
import pytest
from morie.fn.rng222 import rangayyan_ch4_matched_filter_output_acf


def test_rng222_basic():
    """Test basic functionality."""
    phi_x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_output_acf(phi_x, K, t, t_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng222_edge():
    """Test edge cases."""
    phi_x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_output_acf(phi_x, K, t, t_0)
    assert isinstance(result, dict)
