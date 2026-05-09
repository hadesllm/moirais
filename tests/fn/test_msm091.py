"""Tests for msm091.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from moirais.fn.msm091 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm091_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    code = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    reproduce = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(The, R, code, to, reproduce, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm091_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    code = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    reproduce = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(The, R, code, to, reproduce, the)
    assert isinstance(result, dict)
