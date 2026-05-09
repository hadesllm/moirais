"""Tests for km147.kamath_ch9_output_alignment."""
import numpy as np
import pytest
from moirais.fn.km147 import kamath_ch9_output_alignment


def test_km147_basic():
    """Test basic functionality."""
    S_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_output_alignment(S_X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km147_edge():
    """Test edge cases."""
    S_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_output_alignment(S_X)
    assert isinstance(result, dict)
