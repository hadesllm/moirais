"""Tests for msm125.mvsml_categorical_count_eq_8_2."""
import numpy as np
import pytest
from moirais.fn.msm125 import mvsml_categorical_count_eq_8_2


def test_msm125_basic():
    """Test basic functionality."""
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    admits = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_2(dimensional, space, the, solution, admits, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm125_edge():
    """Test edge cases."""
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    admits = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_2(dimensional, space, the, solution, admits, a)
    assert isinstance(result, dict)
