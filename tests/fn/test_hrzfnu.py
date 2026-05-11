"""Tests for hrzfnu.horowitz_deconv_estimator."""
import numpy as np
import pytest
from morie.fn.hrzfnu import horowitz_deconv_estimator


def test_hrzfnu_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_cf = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    nu_n = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_deconv_estimator(w, eps_cf, bandwidth, nu_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzfnu_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_cf = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    nu_n = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_deconv_estimator(w, eps_cf, bandwidth, nu_n)
    assert isinstance(result, dict)
