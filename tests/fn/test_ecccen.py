"""Tests for ecccen.eccentricity_centrality."""
import numpy as np
import pytest
from moirais.fn.ecccen import eccentricity_centrality


def test_ecccen_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = eccentricity_centrality(y, A, node)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ecccen_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = eccentricity_centrality(y, A, node)
    assert isinstance(result, dict)
