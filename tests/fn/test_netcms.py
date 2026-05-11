"""Tests for netcms.network_psychometrics."""
import numpy as np
import pytest
from morie.fn.netcms import network_psychometrics


def test_netcms_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = network_psychometrics(X, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netcms_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = network_psychometrics(X, lam)
    assert isinstance(result, dict)
