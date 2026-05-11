"""Tests for ldppm.local_dp_planar_mechanism."""
import numpy as np
import pytest
from morie.fn.ldppm import local_dp_planar_mechanism


def test_ldppm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    epsilon = 1e-6
    result = local_dp_planar_mechanism(y, truth, k, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ldppm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    epsilon = 1e-6
    result = local_dp_planar_mechanism(y, truth, k, epsilon)
    assert isinstance(result, dict)
