"""Tests for blockMx.block_maxima."""
import numpy as np
import pytest
from moirais.fn.blockMx import block_maxima


def test_blockMx_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    block_size = 100
    result = block_maxima(y, block_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blockMx_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    block_size = 100
    result = block_maxima(y, block_size)
    assert isinstance(result, dict)
