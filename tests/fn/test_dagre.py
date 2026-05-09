"""Tests for moirais.fn.dagre."""
import numpy as np
from moirais.fn.dagre import dagre


def test_dagre_smoke():
    rng = np.random.default_rng(42)
    result = dagre(rater1=rng.integers(0, 4, size=30), rater2=rng.integers(0, 4, size=30))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.dagre import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
