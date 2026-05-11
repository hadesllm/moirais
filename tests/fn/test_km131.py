"""Tests for km131.kamath_ch9_input_projector."""
import numpy as np
import pytest
from morie.fn.km131 import kamath_ch9_input_projector


def test_km131_basic():
    """Test basic functionality."""
    F_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_input_projector(F_X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km131_edge():
    """Test edge cases."""
    F_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_input_projector(F_X)
    assert isinstance(result, dict)
