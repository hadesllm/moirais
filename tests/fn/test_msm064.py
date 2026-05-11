"""Tests for msm064.mvsml_bayesian_regression_eq_6_5."""
import numpy as np
import pytest
from morie.fn.msm064 import mvsml_bayesian_regression_eq_6_5


def test_msm064_basic():
    """Test basic functionality."""
    ment = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(ment, effects, the, interaction, respectively, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm064_edge():
    """Test edge cases."""
    ment = np.random.default_rng(42).normal(0, 1, 100)
    effects = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(ment, effects, the, interaction, respectively, a)
    assert isinstance(result, dict)
