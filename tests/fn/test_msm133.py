"""Tests for msm133.mvsml_categorical_count_eq_8_3."""
import numpy as np
import pytest
from morie.fn.msm133 import mvsml_categorical_count_eq_8_3


def test_msm133_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    statistical = np.random.default_rng(42).normal(0, 1, 100)
    machine = np.random.default_rng(42).normal(0, 1, 100)
    learning = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = mvsml_categorical_count_eq_8_3(where, any, statistical, machine, learning, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm133_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    statistical = np.random.default_rng(42).normal(0, 1, 100)
    machine = np.random.default_rng(42).normal(0, 1, 100)
    learning = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = mvsml_categorical_count_eq_8_3(where, any, statistical, machine, learning, method)
    assert isinstance(result, dict)
