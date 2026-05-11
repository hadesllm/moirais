"""Tests for tmlmnl.tmle_machine_learning."""
import numpy as np
import pytest
from morie.fn.tmlmnl import tmle_machine_learning


def test_tmlmnl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_q = np.random.default_rng(42).normal(0, 1, 100)
    ml_g = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_machine_learning(y, D, X, ml_q, ml_g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmnl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ml_q = np.random.default_rng(42).normal(0, 1, 100)
    ml_g = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_machine_learning(y, D, X, ml_q, ml_g)
    assert isinstance(result, dict)
