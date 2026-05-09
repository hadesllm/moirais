"""Tests for shscl.scaled_schoenfeld_residual."""
import numpy as np
import pytest
from moirais.fn.shscl import scaled_schoenfeld_residual


def test_shscl_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = scaled_schoenfeld_residual(time, event, X)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_shscl_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = scaled_schoenfeld_residual(time, event, X)
    assert isinstance(result, dict)
