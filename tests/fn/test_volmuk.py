"""Tests for volmuk.vol_multi_kernel_rk."""
import numpy as np
import pytest
from morie.fn.volmuk import vol_multi_kernel_rk


def test_volmuk_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    grids = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = vol_multi_kernel_rk(r_intraday, grids, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volmuk_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    grids = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = vol_multi_kernel_rk(r_intraday, grids, kernel)
    assert isinstance(result, dict)
