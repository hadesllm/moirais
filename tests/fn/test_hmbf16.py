"""Tests for hmbf16.geron_bf16."""
import numpy as np
import pytest
from moirais.fn.hmbf16 import geron_bf16


def test_hmbf16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bf16(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbf16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bf16(x)
    assert isinstance(result, dict)
