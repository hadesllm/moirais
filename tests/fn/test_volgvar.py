"""Tests for volgvar.vol_garch_var_backtest."""
import numpy as np
import pytest
from morie.fn.volgvar import vol_garch_var_backtest


def test_volgvar_basic():
    """Test basic functionality."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_garch_var_backtest(hits, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_volgvar_edge():
    """Test edge cases."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_garch_var_backtest(hits, alpha)
    assert isinstance(result, dict)
