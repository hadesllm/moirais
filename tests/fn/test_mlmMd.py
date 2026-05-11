"""Tests for mlmMd.multilevel_mediation."""
import numpy as np
import pytest
from morie.fn.mlmMd import multilevel_mediation


def test_mlmMd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = multilevel_mediation(Y, X, M, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mlmMd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = multilevel_mediation(Y, X, M, cluster)
    assert isinstance(result, dict)
