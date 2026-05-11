"""Tests for msm156.mvsml_categorical_count_eq_8_12."""
import numpy as np
import pytest
from morie.fn.msm156 import mvsml_categorical_count_eq_8_12


def test_msm156_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    Km = np.random.default_rng(42).normal(0, 1, 100)
    nUS21 = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    normal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(where, P, Km, nUS21, a, normal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm156_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    Km = np.random.default_rng(42).normal(0, 1, 100)
    nUS21 = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    normal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(where, P, Km, nUS21, a, normal)
    assert isinstance(result, dict)
