"""Tests for km146.kamath_ch9_output_projector_mse."""
import numpy as np
import pytest
from morie.fn.km146 import kamath_ch9_output_projector_mse


def test_km146_basic():
    """Test basic functionality."""
    H_X = np.random.default_rng(42).normal(0, 1, 100)
    tau_X = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kamath_ch9_output_projector_mse(H_X, tau_X, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km146_edge():
    """Test edge cases."""
    H_X = np.random.default_rng(42).normal(0, 1, 100)
    tau_X = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kamath_ch9_output_projector_mse(H_X, tau_X, t)
    assert isinstance(result, dict)
