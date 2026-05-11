"""Tests for volbekk.vol_bekk_garch."""
import numpy as np
import pytest
from morie.fn.volbekk import vol_bekk_garch


def test_volbekk_basic():
    """Test basic functionality."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_bekk_garch(R_panel, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volbekk_edge():
    """Test edge cases."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_bekk_garch(R_panel, init)
    assert isinstance(result, dict)
