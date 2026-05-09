"""Tests for km130.kamath_ch9_input_alignment_loss."""
import numpy as np
import pytest
from moirais.fn.km130 import kamath_ch9_input_alignment_loss


def test_km130_basic():
    """Test basic functionality."""
    P_X = np.random.default_rng(42).normal(0, 1, 100)
    F_T = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kamath_ch9_input_alignment_loss(P_X, F_T, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km130_edge():
    """Test edge cases."""
    P_X = np.random.default_rng(42).normal(0, 1, 100)
    F_T = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kamath_ch9_input_alignment_loss(P_X, F_T, t)
    assert isinstance(result, dict)
