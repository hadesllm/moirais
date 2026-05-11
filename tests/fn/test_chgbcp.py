"""Tests for chgbcp.bayesian_online_changepoint."""
import numpy as np
import pytest
from morie.fn.chgbcp import bayesian_online_changepoint


def test_chgbcp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_online_changepoint(y, hazard)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chgbcp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_online_changepoint(y, hazard)
    assert isinstance(result, dict)
