"""Tests for eslgbm.esl_gbm."""
import numpy as np
import pytest
from morie.fn.eslgbm import esl_gbm


def test_eslgbm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_gbm(X, y, M, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslgbm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_gbm(X, y, M, nu)
    assert isinstance(result, dict)
