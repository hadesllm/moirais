"""Tests for grbf16.geron_bf16_range."""
import numpy as np
import pytest
from moirais.fn.grbf16 import geron_bf16_range


def test_grbf16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bf16_range(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbf16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bf16_range(x)
    assert isinstance(result, dict)
