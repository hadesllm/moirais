"""Tests for moirais.fn.clust."""
import numpy as np
from moirais.fn.clust import clust


def test_clust_smoke():
    rng = np.random.default_rng(42)
    result = clust(adj_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.clust import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
