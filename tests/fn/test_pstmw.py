"""Tests for morie.fn.pstmw."""
import numpy as np
from morie.fn.pstmw import pstmw


def test_pstmw_smoke():
    rng = np.random.default_rng(42)
    result = pstmw(
        ps=rng.uniform(0.1, 0.9, size=30),
        treatment=rng.integers(0, 2, size=30).astype(float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.pstmw import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
