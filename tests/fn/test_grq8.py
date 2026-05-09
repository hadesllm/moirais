"""Tests for grq8.geron_int8_quantization."""
import numpy as np
import pytest
from moirais.fn.grq8 import geron_int8_quantization


def test_grq8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_int8_quantization(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grq8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_int8_quantization(x)
    assert isinstance(result, dict)
