"""Tests for morie.fn.imhst."""
import numpy as np
from morie.fn.imhst import imhst


def test_imhst_smoke():
    rng = np.random.default_rng(42)
    result = imhst(image=rng.uniform(size=(32, 32)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.imhst import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
