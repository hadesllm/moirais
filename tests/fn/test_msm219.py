"""Tests for msm219.mvsml_ridge_lasso_elastic_eq_9_35."""
import numpy as np
import pytest
from morie.fn.msm219 import mvsml_ridge_lasso_elastic_eq_9_35


def test_msm219_basic():
    """Test basic functionality."""
    labeled = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    correctly = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    ed = np.random.default_rng(42).normal(0, 1, 100)
    those = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_35(labeled, are, correctly, classi, ed, those)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm219_edge():
    """Test edge cases."""
    labeled = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    correctly = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    ed = np.random.default_rng(42).normal(0, 1, 100)
    those = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_35(labeled, are, correctly, classi, ed, those)
    assert isinstance(result, dict)
