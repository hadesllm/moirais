"""Tests for grqat.geron_quantization_aware_training."""
import numpy as np
import pytest
from morie.fn.grqat import geron_quantization_aware_training


def test_grqat_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_quantization_aware_training(x, s, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grqat_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_quantization_aware_training(x, s, bits)
    assert isinstance(result, dict)
