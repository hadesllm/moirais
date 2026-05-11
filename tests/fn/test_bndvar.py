"""Tests for bndvar.bound_variance_term."""
import numpy as np
import pytest
from morie.fn.bndvar import bound_variance_term


def test_bndvar_basic():
    """Test basic functionality."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_variance_term(theta, moments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndvar_edge():
    """Test edge cases."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_variance_term(theta, moments)
    assert isinstance(result, dict)
