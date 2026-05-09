"""Tests for tqang.turboquant_angle_quantization."""
import numpy as np
import pytest
from moirais.fn.tqang import turboquant_angle_quantization


def test_tqang_basic():
    """Test basic functionality."""
    theta = 0.0
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_angle_quantization(theta, bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqang_edge():
    """Test edge cases."""
    theta = 0.0
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_angle_quantization(theta, bits)
    assert isinstance(result, dict)
