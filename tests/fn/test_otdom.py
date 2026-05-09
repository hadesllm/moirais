"""Tests for otdom.ot_domain_adaptation."""
import numpy as np
import pytest
from moirais.fn.otdom import ot_domain_adaptation


def test_otdom_basic():
    """Test basic functionality."""
    Xs = np.random.default_rng(42).normal(0, 1, 100)
    Xt = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_domain_adaptation(Xs, Xt, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otdom_edge():
    """Test edge cases."""
    Xs = np.random.default_rng(42).normal(0, 1, 100)
    Xt = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_domain_adaptation(Xs, Xt, epsilon)
    assert isinstance(result, dict)
