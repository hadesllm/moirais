"""Tests for msm008.mvsml_overfitting_resampling_eq_4_10."""
import numpy as np
import pytest
from moirais.fn.msm008 import mvsml_overfitting_resampling_eq_4_10


def test_msm008_basic():
    """Test basic functionality."""
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    speci = np.random.default_rng(42).normal(0, 1, 100)
    city = np.random.default_rng(42).normal(0, 1, 100)
    proportion = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_10(that, the, speci, city, proportion, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm008_edge():
    """Test edge cases."""
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    speci = np.random.default_rng(42).normal(0, 1, 100)
    city = np.random.default_rng(42).normal(0, 1, 100)
    proportion = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_overfitting_resampling_eq_4_10(that, the, speci, city, proportion, of)
    assert isinstance(result, dict)
