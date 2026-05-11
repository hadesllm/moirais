"""Tests for morie.fn.dgrds."""
import numpy as np
from morie.fn.dgrds import dgrds


def test_dgrds_smoke():
    rng = np.random.default_rng(42)
    result = dgrds(adj_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.dgrds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
