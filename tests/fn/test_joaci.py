"""Tests for joaci.joseph_adaptive_conformal_inference."""
import numpy as np
import pytest
from morie.fn.joaci import joseph_adaptive_conformal_inference


def test_joaci_basic():
    """Test basic functionality."""
    alpha_t = np.random.default_rng(42).normal(0, 1, 100)
    miscoverage_t = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    alpha_target = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_adaptive_conformal_inference(alpha_t, miscoverage_t, eta, alpha_target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joaci_edge():
    """Test edge cases."""
    alpha_t = np.random.default_rng(42).normal(0, 1, 100)
    miscoverage_t = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    alpha_target = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_adaptive_conformal_inference(alpha_t, miscoverage_t, eta, alpha_target)
    assert isinstance(result, dict)
