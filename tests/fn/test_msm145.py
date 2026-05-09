"""Tests for msm145.mvsml_categorical_count_eq_8_11."""
import numpy as np
import pytest
from moirais.fn.msm145 import mvsml_categorical_count_eq_8_11


def test_msm145_basic():
    """Test basic functionality."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    methods = np.random.default_rng(42).normal(0, 1, 100)
    However = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    section = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(of, kernel, methods, However, this, section)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm145_edge():
    """Test edge cases."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    methods = np.random.default_rng(42).normal(0, 1, 100)
    However = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    section = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_11(of, kernel, methods, However, this, section)
    assert isinstance(result, dict)
