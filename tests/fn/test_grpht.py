"""Tests for moirais.fn.grpht."""
import numpy as np
from moirais.fn.grpht import graph_from_edges


def test_grpht_smoke():
    rng = np.random.default_rng(42)
    result = graph_from_edges(edges=[(0, 1), (1, 2), (0, 2)])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.grpht import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
