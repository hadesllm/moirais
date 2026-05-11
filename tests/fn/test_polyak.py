"""Tests for polyak.polyak_target."""
import numpy as np
import pytest
from morie.fn.polyak import polyak_target


def test_polyak_basic():
    """Test basic functionality."""
    theta = 0.0
    theta_target = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = polyak_target(theta, theta_target, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_polyak_edge():
    """Test edge cases."""
    theta = 0.0
    theta_target = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = polyak_target(theta, theta_target, tau)
    assert isinstance(result, dict)
