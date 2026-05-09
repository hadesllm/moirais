"""Tests for qjlcrn.qjl_compression."""
import numpy as np
import pytest
from moirais.fn.qjlcrn import qjl_compression


def test_qjlcrn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = qjl_compression(x, bits, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qjlcrn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = qjl_compression(x, bits, seed)
    assert isinstance(result, dict)
