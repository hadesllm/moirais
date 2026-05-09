"""Tests for moirais.fn.btwns."""
import numpy as np
from moirais.fn.btwns import btwns


def test_btwns_smoke():
    rng = np.random.default_rng(42)
    result = btwns(adj_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.btwns import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
