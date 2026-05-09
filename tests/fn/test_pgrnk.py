"""Tests for moirais.fn.pgrnk."""
import numpy as np
from moirais.fn.pgrnk import pgrnk


def test_pgrnk_smoke():
    rng = np.random.default_rng(42)
    result = pgrnk(adj_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.pgrnk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
