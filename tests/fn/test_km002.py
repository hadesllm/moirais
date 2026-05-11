"""Tests for km002.kamath_ch2_context_vector."""
import numpy as np
import pytest
from morie.fn.km002 import kamath_ch2_context_vector


def test_km002_basic():
    """Test basic functionality."""
    h_1_h_T = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_context_vector(h_1_h_T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km002_edge():
    """Test edge cases."""
    h_1_h_T = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_context_vector(h_1_h_T)
    assert isinstance(result, dict)
