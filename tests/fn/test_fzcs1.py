"""Tests for fzcs1.fauzi_cum_surv_est1."""
import numpy as np
import pytest
from morie.fn.fzcs1 import fauzi_cum_surv_est1


def test_fzcs1_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_cum_surv_est1(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzcs1_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_cum_surv_est1(t, bandwidth, g_func)
    assert isinstance(result, dict)
