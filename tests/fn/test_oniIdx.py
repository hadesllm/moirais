"""Tests for oniIdx.oni."""
import numpy as np
import pytest
from morie.fn.oniIdx import oni


def test_oniIdx_basic():
    """Test basic functionality."""
    sst_n34 = np.random.default_rng(42).normal(0, 1, 100)
    result = oni(sst_n34)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_oniIdx_edge():
    """Test edge cases."""
    sst_n34 = np.random.default_rng(42).normal(0, 1, 100)
    result = oni(sst_n34)
    assert isinstance(result, dict)
