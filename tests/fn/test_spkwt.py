"""Tests for spkwt.schabenberger_kriging_weights."""
import numpy as np
import pytest
from morie.fn.spkwt import schabenberger_kriging_weights


def test_spkwt_basic():
    """Test basic functionality."""
    cov_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    cov_target = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_kriging_weights(cov_matrix, cov_target, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkwt_edge():
    """Test edge cases."""
    cov_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    cov_target = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_kriging_weights(cov_matrix, cov_target, coords)
    assert isinstance(result, dict)
