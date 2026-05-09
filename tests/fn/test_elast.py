"""Tests for moirais.fn.elast."""
import numpy as np
from moirais.fn.elast import elasticity


def test_elast_smoke():
    rng = np.random.default_rng(42)
    result = elasticity(
        price=rng.uniform(10, 100, size=30),
        quantity=rng.uniform(10, 100, size=30)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.elast import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
