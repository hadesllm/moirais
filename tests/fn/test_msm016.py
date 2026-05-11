"""Tests for msm016.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from morie.fn.msm016 import mvsml_linear_mixed_models_eq_5_3


def test_msm016_basic():
    """Test basic functionality."""
    genotypic = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    lines = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = mvsml_linear_mixed_models_eq_5_3(genotypic, effects, of, J, lines, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm016_edge():
    """Test edge cases."""
    genotypic = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    lines = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = mvsml_linear_mixed_models_eq_5_3(genotypic, effects, of, J, lines, Z)
    assert isinstance(result, dict)
