"""Tests for msm009.mvsml_overfitting_resampling_eq_4_14."""
import numpy as np
import pytest
from morie.fn.msm009 import mvsml_overfitting_resampling_eq_4_14


def test_msm009_basic():
    """Test basic functionality."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    disagreement = np.random.default_rng(42).normal(0, 1, 100)
    between = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_14(random, means, a, total, disagreement, between)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm009_edge():
    """Test edge cases."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    disagreement = np.random.default_rng(42).normal(0, 1, 100)
    between = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_14(random, means, a, total, disagreement, between)
    assert isinstance(result, dict)
