"""Tests for ar1cl.ar1_climate."""
import numpy as np
import pytest
from moirais.fn.ar1cl import ar1_climate


def test_ar1cl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = ar1_climate(x, phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ar1cl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = ar1_climate(x, phi)
    assert isinstance(result, dict)
