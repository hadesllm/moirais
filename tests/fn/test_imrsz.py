"""Tests for moirais.fn.imrsz."""
import numpy as np
from moirais.fn.imrsz import imrsz


def test_imrsz_smoke():
    rng = np.random.default_rng(42)
    result = imrsz(image=rng.uniform(size=(32, 32)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.imrsz import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
