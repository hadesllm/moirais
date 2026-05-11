"""Tests for msm139.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from morie.fn.msm139 import mvsml_categorical_count_eq_8_8


def test_msm139_basic():
    """Test basic functionality."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    induced = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(e, v, S, the, induced, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm139_edge():
    """Test edge cases."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    induced = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(e, v, S, the, induced, priors)
    assert isinstance(result, dict)
