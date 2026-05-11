"""Tests for morie.fn.bnpar -- Dirichlet process density."""

import numpy as np
from morie.fn.bnpar import dp_density


def test_returns_dict():
    data = np.random.default_rng(42).standard_normal(50)
    result = dp_density(data, n_iter=50)
    assert isinstance(result, dict)
    assert "n_clusters" in result


def test_n_clusters_positive():
    data = np.random.default_rng(42).standard_normal(30)
    result = dp_density(data, n_iter=50)
    assert result["n_clusters"] >= 1


def test_all_assigned():
    data = np.random.default_rng(42).standard_normal(20)
    result = dp_density(data, n_iter=50)
    assert len(result["assignments"]) == 20
