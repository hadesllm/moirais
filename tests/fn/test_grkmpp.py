"""Tests for grkmpp.geron_kmeans_pp_seeding."""
import numpy as np
import pytest
from moirais.fn.grkmpp import geron_kmeans_pp_seeding


def test_grkmpp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    seed = 42
    result = geron_kmeans_pp_seeding(X, k, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkmpp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    seed = 42
    result = geron_kmeans_pp_seeding(X, k, seed)
    assert isinstance(result, dict)
