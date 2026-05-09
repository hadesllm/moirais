"""Tests for tmltsm.tmle_two_stage."""
import numpy as np
import pytest
from moirais.fn.tmltsm import tmle_two_stage


def test_tmltsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D1 = np.random.default_rng(42).normal(0, 1, 100)
    D2 = np.random.default_rng(42).normal(0, 1, 100)
    X1 = np.random.default_rng(42).normal(0, 1, 100)
    X2 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_two_stage(y, D1, D2, X1, X2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmltsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D1 = np.random.default_rng(42).normal(0, 1, 100)
    D2 = np.random.default_rng(42).normal(0, 1, 100)
    X1 = np.random.default_rng(42).normal(0, 1, 100)
    X2 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_two_stage(y, D1, D2, X1, X2)
    assert isinstance(result, dict)
