"""Tests for mleth.mle_theta_estimator."""
import numpy as np
import pytest
from morie.fn.mleth import mle_theta_estimator


def test_mleth_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = mle_theta_estimator(y, P_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mleth_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = mle_theta_estimator(y, P_theta)
    assert isinstance(result, dict)
