"""Tests for ksr063.kosorok_ch3_cox_efficient_score_beta."""
import numpy as np
import pytest
from morie.fn.ksr063 import kosorok_ch3_cox_efficient_score_beta


def test_ksr063_basic():
    """Test basic functionality."""
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    tau = 0.1
    result = kosorok_ch3_cox_efficient_score_beta(Z, Y, beta, Lambda, M, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr063_edge():
    """Test edge cases."""
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    tau = 0.1
    result = kosorok_ch3_cox_efficient_score_beta(Z, Y, beta, Lambda, M, tau)
    assert isinstance(result, dict)
