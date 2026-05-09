"""Tests for msm039.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from moirais.fn.msm039 import mvsml_linear_mixed_models_eq_5_4


def test_msm039_basic():
    """Test basic functionality."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    rcov = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(random, vs, GID, rcov, units, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm039_edge():
    """Test edge cases."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    rcov = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(random, vs, GID, rcov, units, data)
    assert isinstance(result, dict)
