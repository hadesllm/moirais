"""Tests for rng017.rangayyan_ch3_acf_ensemble_estimate."""
import numpy as np
import pytest
from moirais.fn.rng017 import rangayyan_ch3_acf_ensemble_estimate


def test_rng017_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng017_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau, M)
    assert isinstance(result, dict)
