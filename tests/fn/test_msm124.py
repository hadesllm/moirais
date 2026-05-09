"""Tests for msm124.mvsml_categorical_count_eq_8_1."""
import numpy as np
import pytest
from moirais.fn.msm124 import mvsml_categorical_count_eq_8_1


def test_msm124_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    square = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(f, H, the, square, of, norm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm124_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    square = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_1(f, H, the, square, of, norm)
    assert isinstance(result, dict)
