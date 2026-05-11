"""Tests for msm017.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from morie.fn.msm017 import mvsml_linear_mixed_models_eq_5_3


def test_msm017_basic():
    """Test basic functionality."""
    Illustrative = np.random.default_rng(42).normal(0, 1, 100)
    Examples = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Univariate = np.random.default_rng(42).normal(0, 1, 100)
    LMM = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(Illustrative, Examples, of, the, Univariate, LMM)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm017_edge():
    """Test edge cases."""
    Illustrative = np.random.default_rng(42).normal(0, 1, 100)
    Examples = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Univariate = np.random.default_rng(42).normal(0, 1, 100)
    LMM = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(Illustrative, Examples, of, the, Univariate, LMM)
    assert isinstance(result, dict)
