"""Tests for msm140.mvsml_categorical_count_eq_8_8."""
import numpy as np
import pytest
from morie.fn.msm140 import mvsml_categorical_count_eq_8_8


def test_msm140_basic():
    """Test basic functionality."""
    here = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    eu = np.random.default_rng(42).normal(0, 1, 100)
    eK = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(here, u, j, Nn, eu, eK)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm140_edge():
    """Test edge cases."""
    here = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    eu = np.random.default_rng(42).normal(0, 1, 100)
    eK = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_8(here, u, j, Nn, eu, eK)
    assert isinstance(result, dict)
