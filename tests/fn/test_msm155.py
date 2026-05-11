"""Tests for msm155.mvsml_categorical_count_eq_8_12."""
import numpy as np
import pytest
from morie.fn.msm155 import mvsml_categorical_count_eq_8_12


def test_msm155_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Pf = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(y, Pf, Model, similar, to, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm155_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Pf = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_12(y, Pf, Model, similar, to, model)
    assert isinstance(result, dict)
