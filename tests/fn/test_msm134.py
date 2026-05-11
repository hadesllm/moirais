"""Tests for msm134.mvsml_categorical_count_eq_8_3."""
import numpy as np
import pytest
from morie.fn.msm134 import mvsml_categorical_count_eq_8_3


def test_msm134_basic():
    """Test basic functionality."""
    function = np.random.default_rng(42).normal(0, 1, 100)
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(function, It, important, to, point, out)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm134_edge():
    """Test edge cases."""
    function = np.random.default_rng(42).normal(0, 1, 100)
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_3(function, It, important, to, point, out)
    assert isinstance(result, dict)
