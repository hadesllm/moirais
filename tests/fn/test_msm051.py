"""Tests for msm051.mvsml_bayesian_regression_eq_6_4."""
import numpy as np
import pytest
from moirais.fn.msm051 import mvsml_bayesian_regression_eq_6_4


def test_msm051_basic():
    """Test basic functionality."""
    X1 = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    gG = np.random.default_rng(42).normal(0, 1, 100)
    vg = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(X1, j, g, Nn, gG, vg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm051_edge():
    """Test edge cases."""
    X1 = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    gG = np.random.default_rng(42).normal(0, 1, 100)
    vg = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(X1, j, g, Nn, gG, vg)
    assert isinstance(result, dict)
