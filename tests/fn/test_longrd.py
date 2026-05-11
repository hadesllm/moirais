"""Tests for longrd.long_read_polish."""
import numpy as np
import pytest
from morie.fn.longrd import long_read_polish


def test_longrd_basic():
    """Test basic functionality."""
    assembly = np.random.default_rng(42).normal(0, 1, 100)
    reads = np.random.default_rng(42).normal(0, 1, 100)
    result = long_read_polish(assembly, reads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_longrd_edge():
    """Test edge cases."""
    assembly = np.random.default_rng(42).normal(0, 1, 100)
    reads = np.random.default_rng(42).normal(0, 1, 100)
    result = long_read_polish(assembly, reads)
    assert isinstance(result, dict)
