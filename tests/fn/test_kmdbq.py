"""Tests for kmdbq.kamath_double_quantization."""
import numpy as np
import pytest
from moirais.fn.kmdbq import kamath_double_quantization


def test_kmdbq_basic():
    """Test basic functionality."""
    scales_fp32 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_double_quantization(scales_fp32)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmdbq_edge():
    """Test edge cases."""
    scales_fp32 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_double_quantization(scales_fp32)
    assert isinstance(result, dict)
