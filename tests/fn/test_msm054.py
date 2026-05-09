"""Tests for msm054.mvsml_bayesian_regression_eq_6_4."""
import numpy as np
import pytest
from moirais.fn.msm054 import mvsml_bayesian_regression_eq_6_4


def test_msm054_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    GBLUP = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    equivalently = np.random.default_rng(42).normal(0, 1, 100)
    expressed = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(The, GBLUP, can, be, equivalently, expressed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm054_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    GBLUP = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    equivalently = np.random.default_rng(42).normal(0, 1, 100)
    expressed = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(The, GBLUP, can, be, equivalently, expressed)
    assert isinstance(result, dict)
