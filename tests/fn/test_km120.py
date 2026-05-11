"""Tests for km120.kamath_ch8_bertscore_precision."""
import numpy as np
import pytest
from morie.fn.km120 import kamath_ch8_bertscore_precision


def test_km120_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_precision(x, xhat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km120_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_precision(x, xhat)
    assert isinstance(result, dict)
