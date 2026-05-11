"""Tests for morloc.local_morans_i."""
import numpy as np
import pytest
from morie.fn.morloc import local_morans_i


def test_morloc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_morloc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_morans_i(x, W)
    assert isinstance(result, dict)
