"""Tests for shfrm.shared_frailty_marginal."""
import numpy as np
import pytest
from moirais.fn.shfrm import shared_frailty_marginal


def test_shfrm_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = shared_frailty_marginal(time, event, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shfrm_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = shared_frailty_marginal(time, event, X, cluster)
    assert isinstance(result, dict)
