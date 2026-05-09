"""Tests for expmc.exponential_mechanism."""
import numpy as np
import pytest
from moirais.fn.expmc import exponential_mechanism


def test_expmc_basic():
    """Test basic functionality."""
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = exponential_mechanism(candidates, utility, sensitivity, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_expmc_edge():
    """Test edge cases."""
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = exponential_mechanism(candidates, utility, sensitivity, epsilon)
    assert isinstance(result, dict)
