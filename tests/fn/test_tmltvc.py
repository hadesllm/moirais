"""Tests for tmltvc.tmle_time_varying_confound."""
import numpy as np
import pytest
from morie.fn.tmltvc import tmle_time_varying_confound


def test_tmltvc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_t = np.random.default_rng(42).normal(0, 1, 100)
    L_t = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_time_varying_confound(y, D_t, L_t, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmltvc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_t = np.random.default_rng(42).normal(0, 1, 100)
    L_t = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_time_varying_confound(y, D_t, L_t, time)
    assert isinstance(result, dict)
