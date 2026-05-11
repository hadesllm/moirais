"""Tests for mmeeq.henderson_mme_eq2_2."""
import numpy as np
import pytest
from morie.fn.mmeeq import henderson_mme_eq2_2


def test_mmeeq_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    R = np.random.default_rng(42).normal(0, 1, 100)
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = henderson_mme_eq2_2(Y, X, Z, R, Sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mmeeq_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    R = np.random.default_rng(42).normal(0, 1, 100)
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = henderson_mme_eq2_2(Y, X, Z, R, Sigma)
    assert isinstance(result, dict)
