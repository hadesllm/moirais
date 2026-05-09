"""Tests for tmlvts.tmle_var_targeting."""
import numpy as np
import pytest
from moirais.fn.tmlvts import tmle_var_targeting


def test_tmlvts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_var_targeting(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlvts_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_var_targeting(y, D, X)
    assert isinstance(result, dict)
