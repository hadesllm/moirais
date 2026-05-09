"""Tests for moirais.fn.mahal."""
import numpy as np
from moirais.fn.mahal import mahal


def test_mahal_smoke():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 3))
    x = rng.standard_normal(3)
    result = mahal(x=x, data=data)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.mahal import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
