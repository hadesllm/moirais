"""Tests for msm021.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from moirais.fn.msm021 import mvsml_linear_mixed_models_eq_5_4


def test_msm021_basic():
    """Test basic functionality."""
    When = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    has = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    diagonal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(When, E, has, a, non, diagonal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm021_edge():
    """Test edge cases."""
    When = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    has = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    diagonal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(When, E, has, a, non, diagonal)
    assert isinstance(result, dict)
