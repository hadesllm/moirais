"""Tests for thrtmt.threshold_treatment_msm."""
import numpy as np
import pytest
from moirais.fn.thrtmt import threshold_treatment_msm


def test_thrtmt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    threshold_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = threshold_treatment_msm(y, A, W, threshold_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thrtmt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    threshold_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = threshold_treatment_msm(y, A, W, threshold_grid)
    assert isinstance(result, dict)
