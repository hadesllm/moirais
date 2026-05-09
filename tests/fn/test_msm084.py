"""Tests for msm084.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from moirais.fn.msm084 import mvsml_general_eq_1_2


def test_msm084_basic():
    """Test basic functionality."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Code = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Bayesian, Genomic, Linear, Regression, R, Code)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm084_edge():
    """Test edge cases."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Code = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Bayesian, Genomic, Linear, Regression, R, Code)
    assert isinstance(result, dict)
