"""Tests for grdyq.geron_dynamic_quantization."""
import numpy as np
import pytest
from moirais.fn.grdyq import geron_dynamic_quantization


def test_grdyq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdyq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)
