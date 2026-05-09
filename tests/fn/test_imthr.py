"""Tests for moirais.fn.imthr."""
import numpy as np
from moirais.fn.imthr import imthr


def test_imthr_smoke():
    rng = np.random.default_rng(42)
    result = imthr(image=rng.uniform(size=(32, 32)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.imthr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
