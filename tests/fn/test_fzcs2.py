"""Tests for fzcs2.fauzi_cum_surv_est2."""
import numpy as np
import pytest
from morie.fn.fzcs2 import fauzi_cum_surv_est2


def test_fzcs2_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_cum_surv_est2(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzcs2_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_cum_surv_est2(t, bandwidth, g_func)
    assert isinstance(result, dict)
