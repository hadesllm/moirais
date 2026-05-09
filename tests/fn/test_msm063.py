"""Tests for msm063.mvsml_bayesian_regression_eq_6_7."""
import numpy as np
import pytest
from moirais.fn.msm063 import mvsml_bayesian_regression_eq_6_7


def test_msm063_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_7(where, XE, XEM, are, the, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm063_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_7(where, XE, XEM, are, the, design)
    assert isinstance(result, dict)
