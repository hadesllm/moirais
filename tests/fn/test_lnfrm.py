"""Tests for lnfrm.lognormal_frailty."""
import numpy as np
import pytest
from moirais.fn.lnfrm import lognormal_frailty


def test_lnfrm_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = lognormal_frailty(time, event, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lnfrm_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = lognormal_frailty(time, event, X, cluster)
    assert isinstance(result, dict)
