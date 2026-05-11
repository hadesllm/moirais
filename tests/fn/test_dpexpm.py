"""Tests for dpexpm.dp_exponential_mechanism."""
import numpy as np
import pytest
from morie.fn.dpexpm import dp_exponential_mechanism


def test_dpexpm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_exponential_mechanism(y, candidates, utility, epsilon, sensitivity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpexpm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_exponential_mechanism(y, candidates, utility, epsilon, sensitivity)
    assert isinstance(result, dict)
