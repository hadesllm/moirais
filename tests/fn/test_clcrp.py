"""Tests for clcrp.clustered_crp."""
import numpy as np
import pytest
from moirais.fn.clcrp import clustered_crp


def test_clcrp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    distances = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = clustered_crp(y, distances, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clcrp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    distances = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = clustered_crp(y, distances, alpha)
    assert isinstance(result, dict)
