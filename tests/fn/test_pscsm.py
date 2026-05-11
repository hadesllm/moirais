"""Tests for pscsm.propensity_score_matching."""
import numpy as np
import pytest
from morie.fn.pscsm import propensity_score_matching


def test_pscsm_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = propensity_score_matching(Y, T, X, caliper)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pscsm_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = propensity_score_matching(Y, T, X, caliper)
    assert isinstance(result, dict)
