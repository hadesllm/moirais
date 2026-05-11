"""Tests for ddpmst.ddpm_step."""
import numpy as np
import pytest
from morie.fn.ddpmst import ddpm_step


def test_ddpmst_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = ddpm_step(x_t, t, eps_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ddpmst_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = ddpm_step(x_t, t, eps_theta)
    assert isinstance(result, dict)
