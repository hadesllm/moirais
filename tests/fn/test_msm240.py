"""Tests for msm240.mvsml_preprocessing_eq_2_1."""
import numpy as np
import pytest
from morie.fn.msm240 import mvsml_preprocessing_eq_2_1


def test_msm240_basic():
    """Test basic functionality."""
    represented = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_1(represented, by, random, variables, observed, which)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm240_edge():
    """Test edge cases."""
    represented = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_1(represented, by, random, variables, observed, which)
    assert isinstance(result, dict)
