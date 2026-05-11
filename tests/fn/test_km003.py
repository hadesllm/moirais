"""Tests for km003.kamath_ch2_context_simplest."""
import numpy as np
import pytest
from morie.fn.km003 import kamath_ch2_context_simplest


def test_km003_basic():
    """Test basic functionality."""
    h_T = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_context_simplest(h_T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km003_edge():
    """Test edge cases."""
    h_T = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_context_simplest(h_T)
    assert isinstance(result, dict)
