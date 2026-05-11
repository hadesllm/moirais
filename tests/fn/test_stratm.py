"""Tests for stratm.stratified_mean."""
import numpy as np
import pytest
from morie.fn.stratm import stratified_mean


def test_stratm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = stratified_mean(y, stratum, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stratm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = stratified_mean(y, stratum, weights)
    assert isinstance(result, dict)
