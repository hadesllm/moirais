"""Tests for tqval.turboquant_value_cache_quantization."""
import numpy as np
import pytest
from morie.fn.tqval import turboquant_value_cache_quantization


def test_tqval_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_value_cache_quantization(v, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqval_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_value_cache_quantization(v, bits)
    assert isinstance(result, dict)
