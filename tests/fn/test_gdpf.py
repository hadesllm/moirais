"""Tests for gdpf.gaussian_dp."""
import numpy as np
import pytest
from moirais.fn.gdpf import gaussian_dp


def test_gdpf_basic():
    """Test basic functionality."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = gaussian_dp(mech, mu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gdpf_edge():
    """Test edge cases."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = gaussian_dp(mech, mu)
    assert isinstance(result, dict)
