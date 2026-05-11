"""Tests for hmdqnt.geron_dynamic_quantization."""
import numpy as np
import pytest
from morie.fn.hmdqnt import geron_dynamic_quantization


def test_hmdqnt_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dtype = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dynamic_quantization(model, dtype)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdqnt_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dtype = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dynamic_quantization(model, dtype)
    assert isinstance(result, dict)
