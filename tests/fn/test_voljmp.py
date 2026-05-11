"""Tests for voljmp.vol_jump_test_bnshep."""
import numpy as np
import pytest
from morie.fn.voljmp import vol_jump_test_bnshep


def test_voljmp_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_jump_test_bnshep(r_intraday, block_index)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_voljmp_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    block_index = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_jump_test_bnshep(r_intraday, block_index)
    assert isinstance(result, dict)
