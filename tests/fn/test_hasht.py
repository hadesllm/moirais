"""Tests for moirais.fn.hasht."""
import numpy as np
from moirais.fn.hasht import hash_table


def test_hasht_smoke():
    rng = np.random.default_rng(42)
    result = hash_table(keys=["alpha", "beta", "gamma"], values=[1, 2, 3])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.hasht import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
