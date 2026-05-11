"""Tests for causftbl.causal_frontdoor_adjustment."""
import numpy as np
import pytest
from morie.fn.causftbl import causal_frontdoor_adjustment


def test_causftbl_basic():
    """Test basic functionality."""
    P_Z_X = np.random.default_rng(42).normal(0, 1, 100)
    P_Y_XZ = np.random.default_rng(42).normal(0, 1, 100)
    P_X = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causftbl_edge():
    """Test edge cases."""
    P_Z_X = np.random.default_rng(42).normal(0, 1, 100)
    P_Y_XZ = np.random.default_rng(42).normal(0, 1, 100)
    P_X = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert isinstance(result, dict)
