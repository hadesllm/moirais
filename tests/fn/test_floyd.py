"""Tests for morie.fn.floyd."""
import numpy as np
from morie.fn.floyd import floyd


def test_floyd_smoke():
    rng = np.random.default_rng(42)
    result = floyd(dist_matrix=np.abs(rng.standard_normal((10, 10))) + np.eye(10))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.floyd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
