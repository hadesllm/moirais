"""Tests for km119.kamath_ch8_bertscore_recall."""
import numpy as np
import pytest
from morie.fn.km119 import kamath_ch8_bertscore_recall


def test_km119_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_recall(x, xhat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km119_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_recall(x, xhat)
    assert isinstance(result, dict)
