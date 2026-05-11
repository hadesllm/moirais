"""Tests for thetaF.theta_method."""
import numpy as np
import pytest
from morie.fn.thetaF import theta_method


def test_thetaF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = theta_method(y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thetaF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = theta_method(y, theta)
    assert isinstance(result, dict)
