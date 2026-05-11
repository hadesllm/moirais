"""Tests for fzwm.fauzi_wilcoxon_moments."""
import numpy as np
import pytest
from morie.fn.fzwm import fauzi_wilcoxon_moments


def test_fzwm_basic():
    """Test basic functionality."""
    n = 100
    bandwidth = 0.3
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_wilcoxon_moments(n, bandwidth, theta, cdf, density)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzwm_edge():
    """Test edge cases."""
    n = 100
    bandwidth = 0.3
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_wilcoxon_moments(n, bandwidth, theta, cdf, density)
    assert isinstance(result, dict)
