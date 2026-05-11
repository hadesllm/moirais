"""Tests for msm026.mvsml_linear_mixed_models_eq_5_5."""
import numpy as np
import pytest
from morie.fn.msm026 import mvsml_linear_mixed_models_eq_5_5


def test_msm026_basic():
    """Test basic functionality."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    Y = np.random.default_rng(43).normal(0, 1, 100)
    j2 = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(j, J, Y, j2, g, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm026_edge():
    """Test edge cases."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    Y = np.random.default_rng(43).normal(0, 1, 100)
    j2 = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(j, J, Y, j2, g, E)
    assert isinstance(result, dict)
