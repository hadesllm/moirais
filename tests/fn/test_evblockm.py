"""Tests for evblockm.evt_block_maxima_fit."""
import numpy as np
import pytest
from morie.fn.evblockm import evt_block_maxima_fit


def test_evblockm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_block_maxima_fit(x, block)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evblockm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_block_maxima_fit(x, block)
    assert isinstance(result, dict)
