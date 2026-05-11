"""Tests for evextsl.evt_extremal_index_slidblk."""
import numpy as np
import pytest
from morie.fn.evextsl import evt_extremal_index_slidblk


def test_evextsl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = evt_extremal_index_slidblk(x, block_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evextsl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = evt_extremal_index_slidblk(x, block_size)
    assert isinstance(result, dict)
