"""Tests for moirais.fn.gapst."""
import numpy as np
from moirais.fn.gapst import gapst


def test_gapst_smoke():
    rng = np.random.default_rng(42)
    result = gapst(X=rng.standard_normal((30, 3)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.gapst import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
