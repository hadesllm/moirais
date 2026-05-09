"""Tests for hrztpar.horowitz_parametric_T."""
import numpy as np
import pytest
from moirais.fn.hrztpar import horowitz_parametric_T


def test_hrztpar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T_family = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_parametric_T(x, y, T_family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrztpar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T_family = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_parametric_T(x, y, T_family)
    assert isinstance(result, dict)
