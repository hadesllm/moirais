"""Tests for volrv.vol_realised_variance."""
import numpy as np
import pytest
from morie.fn.volrv import vol_realised_variance


def test_volrv_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_variance(r_intraday, block_index)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrv_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_variance(r_intraday, block_index)
    assert isinstance(result, dict)
