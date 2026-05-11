"""Tests for msm020.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from morie.fn.msm020 import mvsml_linear_mixed_models_eq_5_4


def test_msm020_basic():
    """Test basic functionality."""
    Ei = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    genetic = np.random.default_rng(42).normal(0, 1, 100)
    variance = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(Ei, the, genetic, variance, environment, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm020_edge():
    """Test edge cases."""
    Ei = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    genetic = np.random.default_rng(42).normal(0, 1, 100)
    variance = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(Ei, the, genetic, variance, environment, i)
    assert isinstance(result, dict)
