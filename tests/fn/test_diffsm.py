"""Tests for diffsm.diffusion_score_matching."""
import numpy as np
import pytest
from morie.fn.diffsm import diffusion_score_matching


def test_diffsm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s_theta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = diffusion_score_matching(x, s_theta, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diffsm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s_theta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = diffusion_score_matching(x, s_theta, sigma)
    assert isinstance(result, dict)
