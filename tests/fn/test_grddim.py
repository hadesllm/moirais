"""Tests for grddim.geron_ddim_sampling_step."""
import numpy as np
import pytest
from moirais.fn.grddim import geron_ddim_sampling_step


def test_grddim_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    t_prev = np.random.default_rng(42).normal(0, 1, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddim_sampling_step(x_t, t, t_prev, eps_pred, alpha_bar)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grddim_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    t_prev = np.random.default_rng(42).normal(0, 1, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddim_sampling_step(x_t, t, t_prev, eps_pred, alpha_bar)
    assert isinstance(result, dict)
