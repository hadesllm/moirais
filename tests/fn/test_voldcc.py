"""Tests for voldcc.vol_dcc_garch."""
import numpy as np
import pytest
from moirais.fn.voldcc import vol_dcc_garch


def test_voldcc_basic():
    """Test basic functionality."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_dcc_garch(R_panel, init)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_voldcc_edge():
    """Test edge cases."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_dcc_garch(R_panel, init)
    assert isinstance(result, dict)
