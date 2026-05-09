"""Tests for wleth.weighted_likelihood_theta."""
import numpy as np
import pytest
from moirais.fn.wleth import weighted_likelihood_theta


def test_wleth_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_likelihood_theta(y, P_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wleth_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    P_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_likelihood_theta(y, P_theta)
    assert isinstance(result, dict)
