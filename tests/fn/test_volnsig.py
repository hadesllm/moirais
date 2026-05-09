"""Tests for volnsig.vol_nelson_skew_garch."""
import numpy as np
import pytest
from moirais.fn.volnsig import vol_nelson_skew_garch


def test_volnsig_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_nelson_skew_garch(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volnsig_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_nelson_skew_garch(r, init)
    assert isinstance(result, dict)
