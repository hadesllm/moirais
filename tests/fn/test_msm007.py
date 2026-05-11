"""Tests for msm007.mvsml_overfitting_resampling_eq_4_9."""
import numpy as np
import pytest
from morie.fn.msm007 import mvsml_overfitting_resampling_eq_4_9


def test_msm007_basic():
    """Test basic functionality."""
    proportion = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    true = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_9(proportion, of, true, positives, that, are)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm007_edge():
    """Test edge cases."""
    proportion = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    true = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_9(proportion, of, true, positives, that, are)
    assert isinstance(result, dict)
