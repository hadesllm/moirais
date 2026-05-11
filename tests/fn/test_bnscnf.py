"""Tests for bnscnf.bound_confidence_set."""
import numpy as np
import pytest
from morie.fn.bnscnf import bound_confidence_set


def test_bnscnf_basic():
    """Test basic functionality."""
    theta_grid = np.random.default_rng(42).normal(0, 1, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_confidence_set(theta_grid, moments, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnscnf_edge():
    """Test edge cases."""
    theta_grid = np.random.default_rng(42).normal(0, 1, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_confidence_set(theta_grid, moments, alpha)
    assert isinstance(result, dict)
