"""Tests for rng215.rangayyan_ch4_snr_normalized_ratio."""
import numpy as np
import pytest
from moirais.fn.rng215 import rangayyan_ch4_snr_normalized_ratio


def test_rng215_basic():
    """Test basic functionality."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_snr_normalized_ratio(H, X, P_eta_i, f, t_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng215_edge():
    """Test edge cases."""
    H = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_snr_normalized_ratio(H, X, P_eta_i, f, t_0)
    assert isinstance(result, dict)
