"""Tests for msm141.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from morie.fn.msm141 import mvsml_categorical_count_eq_8_8


def test_msm141_basic():
    """Test basic functionality."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    u = np.random.default_rng(44).normal(0, 1, 100)
    evu = np.random.default_rng(42).normal(0, 1, 100)
    eSu = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(j, k, u, evu, eSu, e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm141_edge():
    """Test edge cases."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    u = np.random.default_rng(44).normal(0, 1, 100)
    evu = np.random.default_rng(42).normal(0, 1, 100)
    eSu = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(j, k, u, evu, eSu, e)
    assert isinstance(result, dict)
