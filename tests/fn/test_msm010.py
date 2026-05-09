"""Tests for msm010.mvsml_linear_mixed_models_eq_5_1."""
import numpy as np
import pytest
from moirais.fn.msm010 import mvsml_linear_mixed_models_eq_5_1


def test_msm010_basic():
    """Test basic functionality."""
    O = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Montesinos = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    pez = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(O, A, Montesinos, L, pez, et)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm010_edge():
    """Test edge cases."""
    O = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Montesinos = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    pez = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(O, A, Montesinos, L, pez, et)
    assert isinstance(result, dict)
