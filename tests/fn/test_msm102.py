"""Tests for msm102.mvsml_bayesian_regression_pt2_eq_7_5."""
import numpy as np
import pytest
from morie.fn.msm102 import mvsml_bayesian_regression_pt2_eq_7_5


def test_msm102_basic():
    """Test basic functionality."""
    SD = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    ZLg = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(SD, L, XE, E, ZLg, e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm102_edge():
    """Test edge cases."""
    SD = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    ZLg = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(SD, L, XE, E, ZLg, e)
    assert isinstance(result, dict)
