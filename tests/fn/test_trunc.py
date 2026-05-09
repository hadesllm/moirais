"""Tests for moirais.fn.trunc."""
import numpy as np
from moirais.fn.trunc import trunc


def test_trunc_smoke():
    rng = np.random.default_rng(42)
    result = trunc(
        y=rng.uniform(-80, -75, size=20),
        x=rng.uniform(40, 45, size=20),
        threshold=0.0
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.trunc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
