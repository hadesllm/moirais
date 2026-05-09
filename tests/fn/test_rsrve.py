"""Tests for moirais.fn.rsrve."""
import numpy as np
from moirais.fn.rsrve import rsrve


def test_rsrve_smoke():
    rng = np.random.default_rng(42)
    result = rsrve(
        triangle=np.array([[100, 50, 30], [110, 60, 0], [120, 0, 0]], dtype=float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.rsrve import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
