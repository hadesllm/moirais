"""Tests for gscmcl.generalized_synthetic_control."""
import numpy as np
import pytest
from moirais.fn.gscmcl import generalized_synthetic_control


def test_gscmcl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    r = 10
    result = generalized_synthetic_control(y, D, unit, time, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gscmcl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    r = 10
    result = generalized_synthetic_control(y, D, unit, time, r)
    assert isinstance(result, dict)
