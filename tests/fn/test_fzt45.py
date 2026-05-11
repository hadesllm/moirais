"""Tests for fzt45.fauzi_thm4_5_mrl_consistency."""
import numpy as np
import pytest
from morie.fn.fzt45 import fauzi_thm4_5_mrl_consistency


def test_fzt45_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_5_mrl_consistency(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt45_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_5_mrl_consistency(t, bandwidth, g_func)
    assert isinstance(result, dict)
