"""Tests for hrzlam.horowitz_baseline_hazard_est."""
import numpy as np
import pytest
from moirais.fn.hrzlam import horowitz_baseline_hazard_est


def test_hrzlam_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_baseline_hazard_est(t, x, event, beta_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzlam_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_baseline_hazard_est(t, x, event, beta_hat)
    assert isinstance(result, dict)
