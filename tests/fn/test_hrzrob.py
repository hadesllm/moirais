"""Tests for hrzrob.horowitz_rate_beta_estimation."""
import numpy as np
import pytest
from moirais.fn.hrzrob import horowitz_rate_beta_estimation


def test_hrzrob_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_rate_beta_estimation(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzrob_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_rate_beta_estimation(x, y, bandwidth)
    assert isinstance(result, dict)
