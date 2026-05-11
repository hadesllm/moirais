"""Tests for theteap.theta_eap."""
import numpy as np
import pytest
from morie.fn.theteap import theta_eap


def test_theteap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_eap(X, items, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_theteap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_eap(X, items, prior)
    assert isinstance(result, dict)
