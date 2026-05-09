"""Tests for moirais.fn.krus."""
import numpy as np
from moirais.fn.krus import krus


def test_krus_smoke():
    rng = np.random.default_rng(42)
    result = krus(edges=np.array([[0, 1, 1.0], [1, 2, 2.0], [0, 2, 3.0]]), n_nodes=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.krus import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
