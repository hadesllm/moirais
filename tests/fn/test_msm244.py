"""Tests for msm244.mvsml_preprocessing_eq_2_1."""
import numpy as np
import pytest
from moirais.fn.msm244 import mvsml_preprocessing_eq_2_1


def test_msm244_basic():
    """Test basic functionality."""
    In = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    only = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    columns = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_1(In, matrix, M, only, the, columns)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm244_edge():
    """Test edge cases."""
    In = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    only = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    columns = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_preprocessing_eq_2_1(In, matrix, M, only, the, columns)
    assert isinstance(result, dict)
