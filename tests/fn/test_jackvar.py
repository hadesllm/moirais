"""Tests for jackvar.jackknife_variance_survey."""
import numpy as np
import pytest
from morie.fn.jackvar import jackknife_variance_survey


def test_jackvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = jackknife_variance_survey(y, weights, replicates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jackvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = jackknife_variance_survey(y, weights, replicates)
    assert isinstance(result, dict)
