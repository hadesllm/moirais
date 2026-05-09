"""Tests for ksr068.kosorok_ch3_cox_profile_score."""
import numpy as np
import pytest
from moirais.fn.ksr068 import kosorok_ch3_cox_profile_score


def test_ksr068_basic():
    """Test basic functionality."""
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    n = 100
    result = kosorok_ch3_cox_profile_score(beta, Z, Y, X, tau, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr068_edge():
    """Test edge cases."""
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    n = 100
    result = kosorok_ch3_cox_profile_score(beta, Z, Y, X, tau, n)
    assert isinstance(result, dict)
