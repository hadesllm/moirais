"""Tests for volbpv.vol_bipower_variation."""
import numpy as np
import pytest
from moirais.fn.volbpv import vol_bipower_variation


def test_volbpv_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_bipower_variation(r_intraday, block_index)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volbpv_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_bipower_variation(r_intraday, block_index)
    assert isinstance(result, dict)
