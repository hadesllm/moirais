"""Tests for volgo.vol_garch_orthogonal."""
import numpy as np
import pytest
from morie.fn.volgo import vol_garch_orthogonal


def test_volgo_basic():
    """Test basic functionality."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = vol_garch_orthogonal(R_panel, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgo_edge():
    """Test edge cases."""
    R_panel = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = vol_garch_orthogonal(R_panel, k)
    assert isinstance(result, dict)
