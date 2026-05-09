"""Tests for dprrep.randomized_response_dp."""
import numpy as np
import pytest
from moirais.fn.dprrep import randomized_response_dp


def test_dprrep_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = randomized_response_dp(y, truth, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dprrep_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = randomized_response_dp(y, truth, epsilon)
    assert isinstance(result, dict)
