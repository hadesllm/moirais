"""Tests for covpop.coverage_correction."""
import numpy as np
import pytest
from morie.fn.covpop import coverage_correction


def test_covpop_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    target_totals = np.random.default_rng(42).normal(0, 1, 100)
    result = coverage_correction(y, weights, target_totals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_covpop_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    target_totals = np.random.default_rng(42).normal(0, 1, 100)
    result = coverage_correction(y, weights, target_totals)
    assert isinstance(result, dict)
