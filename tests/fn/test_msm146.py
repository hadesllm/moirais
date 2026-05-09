"""Tests for msm146.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from moirais.fn.msm146 import mvsml_categorical_count_eq_8_8


def test_msm146_basic():
    """Test basic functionality."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(it, important, to, point, out, that)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm146_edge():
    """Test edge cases."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(it, important, to, point, out, that)
    assert isinstance(result, dict)
