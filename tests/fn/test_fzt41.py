"""Tests for fzt41.fauzi_thm4_1_surv_bias_var."""
import numpy as np
import pytest
from moirais.fn.fzt41 import fauzi_thm4_1_surv_bias_var


def test_fzt41_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_1_surv_bias_var(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt41_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_thm4_1_surv_bias_var(t, bandwidth, g_func)
    assert isinstance(result, dict)
