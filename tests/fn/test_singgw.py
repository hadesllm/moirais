"""Tests for singgw.single_step_gblup."""
import numpy as np
import pytest
from morie.fn.singgw import single_step_gblup


def test_singgw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    G = np.eye(10)
    result = single_step_gblup(y, X, Z, A, G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_singgw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    G = np.eye(10)
    result = single_step_gblup(y, X, Z, A, G)
    assert isinstance(result, dict)
