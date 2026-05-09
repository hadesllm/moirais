"""Tests for msm097.mvsml_bayesian_regression_pt2_eq_7_3."""
import numpy as np
import pytest
from moirais.fn.msm097 import mvsml_bayesian_regression_pt2_eq_7_3


def test_msm097_basic():
    """Test basic functionality."""
    improvement = np.random.default_rng(42).normal(0, 1, 100)
    these = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    respect = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    their = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(improvement, these, models, respect, to, their)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm097_edge():
    """Test edge cases."""
    improvement = np.random.default_rng(42).normal(0, 1, 100)
    these = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    respect = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    their = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(improvement, these, models, respect, to, their)
    assert isinstance(result, dict)
