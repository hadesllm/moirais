"""Tests for moirais.fn.valfn."""
import numpy as np
from moirais.fn.valfn import valfn


def test_valfn_smoke():
    rng = np.random.default_rng(42)
    result = valfn(returns=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.valfn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
