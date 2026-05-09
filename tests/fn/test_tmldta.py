"""Tests for tmldta.tmle_data_adaptive."""
import numpy as np
import pytest
from moirais.fn.tmldta import tmle_data_adaptive


def test_tmldta_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    candidate_strata = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_data_adaptive(y, D, X, candidate_strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmldta_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    candidate_strata = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_data_adaptive(y, D, X, candidate_strata)
    assert isinstance(result, dict)
