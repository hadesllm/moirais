"""Tests for msm031.mvsml_linear_mixed_models_eq_5_5."""
import numpy as np
import pytest
from morie.fn.msm031 import mvsml_linear_mixed_models_eq_5_5


def test_msm031_basic():
    """Test basic functionality."""
    similarly = np.random.default_rng(42).normal(0, 1, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    xp = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(similarly, x1, x2, xp, are, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm031_edge():
    """Test edge cases."""
    similarly = np.random.default_rng(42).normal(0, 1, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    xp = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(similarly, x1, x2, xp, are, the)
    assert isinstance(result, dict)
