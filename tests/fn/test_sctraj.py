"""Tests for sctraj.scrnaseq_trajectory."""
import numpy as np
import pytest
from moirais.fn.sctraj import scrnaseq_trajectory


def test_sctraj_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    clusters = np.random.default_rng(42).normal(0, 1, 100)
    start_cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = scrnaseq_trajectory(X, clusters, start_cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sctraj_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    clusters = np.random.default_rng(42).normal(0, 1, 100)
    start_cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = scrnaseq_trajectory(X, clusters, start_cluster)
    assert isinstance(result, dict)
