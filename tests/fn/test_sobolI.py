"""Tests for sobolI.sobol_indices."""
import numpy as np
import pytest
from morie.fn.sobolI import sobol_indices


def test_sobolI_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = sobol_indices(model, input_dist, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sobolI_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = sobol_indices(model, input_dist, N)
    assert isinstance(result, dict)
