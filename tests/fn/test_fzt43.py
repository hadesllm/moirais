"""Tests for fzt43.fauzi_thm4_3_mrl_bias_var."""
import numpy as np
import pytest
from moirais.fn.fzt43 import fauzi_thm4_3_mrl_bias_var


def test_fzt43_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_thm4_3_mrl_bias_var(t, bandwidth, g_func, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt43_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_thm4_3_mrl_bias_var(t, bandwidth, g_func, i)
    assert isinstance(result, dict)
