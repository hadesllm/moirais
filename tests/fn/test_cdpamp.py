"""Tests for cdpamp.cdp_subgaussian_amplification."""
import numpy as np
import pytest
from moirais.fn.cdpamp import cdp_subgaussian_amplification


def test_cdpamp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    k_compositions = np.random.default_rng(42).normal(0, 1, 100)
    result = cdp_subgaussian_amplification(y, epsilon, delta, k_compositions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cdpamp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    k_compositions = np.random.default_rng(42).normal(0, 1, 100)
    result = cdp_subgaussian_amplification(y, epsilon, delta, k_compositions)
    assert isinstance(result, dict)
