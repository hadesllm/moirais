"""Tests for moirais.fn.lrpos."""
import numpy as np
from moirais.fn.lrpos import lrpos


def test_lrpos_smoke():
    rng = np.random.default_rng(42)
    result = lrpos(sensitivity=0.9, specificity=0.8)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.lrpos import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
