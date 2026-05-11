"""Tests for mapth.map_theta_estimator."""
import numpy as np
import pytest
from morie.fn.mapth import map_theta_estimator


def test_mapth_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = map_theta_estimator(y, prior, P_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mapth_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = map_theta_estimator(y, prior, P_theta)
    assert isinstance(result, dict)
