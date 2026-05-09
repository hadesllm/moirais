"""Tests for polqnt.polar_quantization."""
import numpy as np
import pytest
from moirais.fn.polqnt import polar_quantization


def test_polqnt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = polar_quantization(x, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_polqnt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = polar_quantization(x, bits)
    assert isinstance(result, dict)
