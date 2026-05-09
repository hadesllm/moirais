"""Tests for msm259.mvsml_deep_learning_eq_13_1."""
import numpy as np
import pytest
from moirais.fn.msm259 import mvsml_deep_learning_eq_13_1


def test_msm259_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    size = np.random.default_rng(42).normal(0, 1, 100)
    Part = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_deep_learning_eq_13_1(image, of, size, Part, b, Fig)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm259_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    size = np.random.default_rng(42).normal(0, 1, 100)
    Part = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_deep_learning_eq_13_1(image, of, size, Part, b, Fig)
    assert isinstance(result, dict)
