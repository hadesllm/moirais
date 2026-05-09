"""Tests for ddimst.ddim_step."""
import numpy as np
import pytest
from moirais.fn.ddimst import ddim_step


def test_ddimst_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_theta = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = ddim_step(x_t, t, eps_theta, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ddimst_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_theta = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = ddim_step(x_t, t, eps_theta, eta)
    assert isinstance(result, dict)
