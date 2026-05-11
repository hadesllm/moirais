"""Tests for msm082.mvsml_bayesian_regression_eq_6_3."""
import numpy as np
import pytest
from morie.fn.msm082 import mvsml_bayesian_regression_eq_6_3


def test_msm082_basic():
    """Test basic functionality."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    jq = np.random.default_rng(42).normal(0, 1, 100)
    Appendix = np.random.default_rng(42).normal(0, 1, 100)
    Setting = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Prior = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(j, jq, Appendix, Setting, the, Prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm082_edge():
    """Test edge cases."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    jq = np.random.default_rng(42).normal(0, 1, 100)
    Appendix = np.random.default_rng(42).normal(0, 1, 100)
    Setting = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Prior = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(j, jq, Appendix, Setting, the, Prior)
    assert isinstance(result, dict)
