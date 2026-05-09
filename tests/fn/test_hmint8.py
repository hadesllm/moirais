"""Tests for hmint8.geron_int8_quant."""
import numpy as np
import pytest
from moirais.fn.hmint8 import geron_int8_quant


def test_hmint8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    symmetric = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_int8_quant(x, n_bits, symmetric)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmint8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    symmetric = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_int8_quant(x, n_bits, symmetric)
    assert isinstance(result, dict)
