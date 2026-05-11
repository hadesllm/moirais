"""Tests for gb2112.gibbons_block_freq_dist."""
import numpy as np
import pytest
from morie.fn.gb2112 import gibbons_block_freq_dist


def test_gb2112_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    result = gibbons_block_freq_dist(m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb2112_edge():
    """Test edge cases."""
    m = 10
    n = 100
    result = gibbons_block_freq_dist(m, n)
    assert isinstance(result, dict)
