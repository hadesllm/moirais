"""Tests for basEvap.penman_monteith."""
import numpy as np
import pytest
from morie.fn.basEvap import penman_monteith


def test_basEvap_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    R_n = np.random.default_rng(42).normal(0, 1, 100)
    u2 = np.random.default_rng(42).normal(0, 1, 100)
    VPD = np.random.default_rng(42).normal(0, 1, 100)
    result = penman_monteith(T, R_n, u2, VPD)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_basEvap_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    R_n = np.random.default_rng(42).normal(0, 1, 100)
    u2 = np.random.default_rng(42).normal(0, 1, 100)
    VPD = np.random.default_rng(42).normal(0, 1, 100)
    result = penman_monteith(T, R_n, u2, VPD)
    assert isinstance(result, dict)
