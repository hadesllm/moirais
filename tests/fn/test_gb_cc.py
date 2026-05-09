"""Tests for gb_cc.gibbons_continuity_corr."""
import numpy as np
import pytest
from moirais.fn.gb_cc import gibbons_continuity_corr


def test_gb_cc_basic():
    """Test basic functionality."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = gibbons_continuity_corr(T, mu, sigma)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_cc_edge():
    """Test edge cases."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = gibbons_continuity_corr(T, mu, sigma)
    assert isinstance(result, dict)
