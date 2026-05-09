"""Tests for moirais.fn.lotka."""
import numpy as np
from moirais.fn.lotka import lotka


def test_lotka_smoke():
    rng = np.random.default_rng(42)
    result = lotka()
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.lotka import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
