"""Tests for msm055.mvsml_bayesian_regression_eq_6_5."""
import numpy as np
import pytest
from morie.fn.msm055 import mvsml_bayesian_regression_eq_6_5


def test_msm055_basic():
    """Test basic functionality."""
    Cholesky = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    genomic = np.random.default_rng(42).normal(0, 1, 100)
    relationship = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(Cholesky, of, the, genomic, relationship, matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm055_edge():
    """Test edge cases."""
    Cholesky = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    genomic = np.random.default_rng(42).normal(0, 1, 100)
    relationship = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(Cholesky, of, the, genomic, relationship, matrix)
    assert isinstance(result, dict)
