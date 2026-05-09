"""Tests for volrk.vol_realised_kernel."""
import numpy as np
import pytest
from moirais.fn.volrk import vol_realised_kernel


def test_volrk_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_kernel(r_intraday, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrk_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_kernel(r_intraday, H)
    assert isinstance(result, dict)
