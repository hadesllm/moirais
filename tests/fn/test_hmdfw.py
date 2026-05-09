"""Tests for hmdfw.geron_diffusion_forward."""
import numpy as np
import pytest
from moirais.fn.hmdfw import geron_diffusion_forward


def test_hmdfw_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    beta_schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_diffusion_forward(x0, T, beta_schedule)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdfw_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    beta_schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_diffusion_forward(x0, T, beta_schedule)
    assert isinstance(result, dict)
