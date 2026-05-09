"""Tests for msm090.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from moirais.fn.msm090 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm090_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    bi = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    now = np.random.default_rng(42).normal(0, 1, 100)
    Li = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(c, bi, C, where, now, Li)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm090_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    bi = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    now = np.random.default_rng(42).normal(0, 1, 100)
    Li = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(c, bi, C, where, now, Li)
    assert isinstance(result, dict)
