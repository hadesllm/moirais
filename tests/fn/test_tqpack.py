"""Tests for tqpack.turboquant_bit_pack_indices."""
import numpy as np
import pytest
from moirais.fn.tqpack import turboquant_bit_pack_indices


def test_tqpack_basic():
    """Test basic functionality."""
    indices = np.arange(0, 100, dtype=int)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_bit_pack_indices(indices, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqpack_edge():
    """Test edge cases."""
    indices = np.arange(0, 100, dtype=int)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_bit_pack_indices(indices, bits)
    assert isinstance(result, dict)
