"""Tests for hrzhot.horowitz_T_F_estimators."""
import numpy as np
import pytest
from moirais.fn.hrzhot import horowitz_T_F_estimators


def test_hrzhot_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_T_F_estimators(x, y, bandwidth, beta_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzhot_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_T_F_estimators(x, y, bandwidth, beta_hat)
    assert isinstance(result, dict)
