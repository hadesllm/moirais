"""Tests for msm126.mvsml_categorical_count_eq_8_2."""
import numpy as np
import pytest
from moirais.fn.msm126 import mvsml_categorical_count_eq_8_2


def test_msm126_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    ned = np.random.default_rng(42).normal(0, 1, 100)
    before = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_2(n, k, k2, de, ned, before)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm126_edge():
    """Test edge cases."""
    n = 100
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    ned = np.random.default_rng(42).normal(0, 1, 100)
    before = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_2(n, k, k2, de, ned, before)
    assert isinstance(result, dict)
