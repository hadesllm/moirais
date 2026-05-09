"""Tests for msm185.mvsml_ridge_lasso_elastic_eq_9_12."""
import numpy as np
import pytest
from moirais.fn.msm185 import mvsml_ridge_lasso_elastic_eq_9_12


def test_msm185_basic():
    """Test basic functionality."""
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    maximize = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_12(optimization, problem, Xm, Xp, maximize, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm185_edge():
    """Test edge cases."""
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    maximize = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_12(optimization, problem, Xm, Xp, maximize, f)
    assert isinstance(result, dict)
