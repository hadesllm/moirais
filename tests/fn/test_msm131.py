"""Tests for msm131.mvsml_categorical_count_eq_8_4."""
import numpy as np
import pytest
from morie.fn.msm131 import mvsml_categorical_count_eq_8_4


def test_msm131_basic():
    """Test basic functionality."""
    de = np.random.default_rng(42).normal(0, 1, 100)
    nite = np.random.default_rng(42).normal(0, 1, 100)
    related = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    ANN = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_4(de, nite, related, to, an, ANN)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm131_edge():
    """Test edge cases."""
    de = np.random.default_rng(42).normal(0, 1, 100)
    nite = np.random.default_rng(42).normal(0, 1, 100)
    related = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    ANN = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_4(de, nite, related, to, an, ANN)
    assert isinstance(result, dict)
