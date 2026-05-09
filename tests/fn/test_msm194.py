"""Tests for msm194.mvsml_ridge_lasso_elastic_eq_9_22."""
import numpy as np
import pytest
from moirais.fn.msm194 import mvsml_ridge_lasso_elastic_eq_9_22


def test_msm194_basic():
    """Test basic functionality."""
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_22(With, this, last, version, of, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm194_edge():
    """Test edge cases."""
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_22(With, this, last, version, of, the)
    assert isinstance(result, dict)
