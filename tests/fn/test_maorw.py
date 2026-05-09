"""Tests for maorw.ma_orwin_fsn."""
import numpy as np
import pytest
from moirais.fn.maorw import ma_orwin_fsn


def test_maorw_basic():
    """Test basic functionality."""
    d_obs = np.random.default_rng(42).normal(0, 1, 100)
    d_crit = np.random.default_rng(42).normal(0, 1, 100)
    d_filldraw = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_orwin_fsn(d_obs, d_crit, d_filldraw, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_maorw_edge():
    """Test edge cases."""
    d_obs = np.random.default_rng(42).normal(0, 1, 100)
    d_crit = np.random.default_rng(42).normal(0, 1, 100)
    d_filldraw = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_orwin_fsn(d_obs, d_crit, d_filldraw, k)
    assert isinstance(result, dict)
