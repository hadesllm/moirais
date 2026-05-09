"""Tests for grdpmr.geron_ddpm_reverse_step."""
import numpy as np
import pytest
from moirais.fn.grdpmr import geron_ddpm_reverse_step


def test_grdpmr_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdpmr_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma)
    assert isinstance(result, dict)
