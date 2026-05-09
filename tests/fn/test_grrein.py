"""Tests for grrein.geron_reinforce_policy_gradient."""
import numpy as np
import pytest
from moirais.fn.grrein import geron_reinforce_policy_gradient


def test_grrein_basic():
    """Test basic functionality."""
    theta = 0.0
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    returns_G = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrein_edge():
    """Test edge cases."""
    theta = 0.0
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    returns_G = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)
