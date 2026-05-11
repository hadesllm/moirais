"""Tests for morie.fn.cmpnt."""
import numpy as np
from morie.fn.cmpnt import cmpnt


def test_cmpnt_smoke():
    rng = np.random.default_rng(42)
    result = cmpnt(adj_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.cmpnt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
