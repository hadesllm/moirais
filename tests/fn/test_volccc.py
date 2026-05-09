"""Tests for volccc.vol_ccc_garch."""
import numpy as np
import pytest
from moirais.fn.volccc import vol_ccc_garch


def test_volccc_basic():
    """Test basic functionality."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_ccc_garch(R_panel, init)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_volccc_edge():
    """Test edge cases."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_ccc_garch(R_panel, init)
    assert isinstance(result, dict)
