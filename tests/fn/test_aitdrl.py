"""Tests for aitdrl.dirichlet_loglik."""
import numpy as np
import pytest
from moirais.fn.aitdrl import dirichlet_loglik


def test_aitdrl_basic():
    """Test basic functionality."""
    alpha = 0.05
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dirichlet_loglik(alpha, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitdrl_edge():
    """Test edge cases."""
    alpha = 0.05
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dirichlet_loglik(alpha, X)
    assert isinstance(result, dict)
