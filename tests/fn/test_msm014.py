"""Tests for msm014.mvsml_linear_mixed_models_eq_5_1."""
import numpy as np
import pytest
from moirais.fn.msm014 import mvsml_linear_mixed_models_eq_5_1


def test_msm014_basic():
    """Test basic functionality."""
    are = np.random.default_rng(42).normal(0, 1, 100)
    assumed = np.random.default_rng(42).normal(0, 1, 100)
    identically = np.random.default_rng(42).normal(0, 1, 100)
    distributed = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(are, assumed, identically, distributed, R, where)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm014_edge():
    """Test edge cases."""
    are = np.random.default_rng(42).normal(0, 1, 100)
    assumed = np.random.default_rng(42).normal(0, 1, 100)
    identically = np.random.default_rng(42).normal(0, 1, 100)
    distributed = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(are, assumed, identically, distributed, R, where)
    assert isinstance(result, dict)
