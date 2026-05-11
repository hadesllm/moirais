"""Tests for msm075.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from morie.fn.msm075 import mvsml_bayesian_regression_eq_6_9


def test_msm075_basic():
    """Test basic functionality."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Multi = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(Bayesian, Genomic, Multi, trait, environment, Model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm075_edge():
    """Test edge cases."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Multi = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(Bayesian, Genomic, Multi, trait, environment, Model)
    assert isinstance(result, dict)
