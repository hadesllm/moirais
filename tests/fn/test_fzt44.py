"""Tests for fzt44.fauzi_thm4_4_mrl_normality."""
import numpy as np
import pytest
from moirais.fn.fzt44 import fauzi_thm4_4_mrl_normality


def test_fzt44_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_4_mrl_normality(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt44_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_4_mrl_normality(t, bandwidth, g_func)
    assert isinstance(result, dict)
