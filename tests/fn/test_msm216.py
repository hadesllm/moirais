"""Tests for msm216.mvsml_ridge_lasso_elastic_eq_9_32."""
import numpy as np
import pytest
from morie.fn.msm216 import mvsml_ridge_lasso_elastic_eq_9_32


def test_msm216_basic():
    """Test basic functionality."""
    denotes = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    dot = np.random.default_rng(42).normal(0, 1, 100)
    product = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    vectors = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(denotes, the, dot, product, of, vectors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm216_edge():
    """Test edge cases."""
    denotes = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    dot = np.random.default_rng(42).normal(0, 1, 100)
    product = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    vectors = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(denotes, the, dot, product, of, vectors)
    assert isinstance(result, dict)
