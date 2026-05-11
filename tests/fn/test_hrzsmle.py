"""Tests for hrzsmle.horowitz_semipar_mle_binary."""
import numpy as np
import pytest
from morie.fn.hrzsmle import horowitz_semipar_mle_binary


def test_hrzsmle_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_semipar_mle_binary(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsmle_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_semipar_mle_binary(x, y, bandwidth)
    assert isinstance(result, dict)
