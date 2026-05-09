"""Tests for moirais.fn.fdiff."""
import numpy as np
from moirais.fn.fdiff import fdiff


def test_fdiff_smoke():
    rng = np.random.default_rng(42)
    result = fdiff(f=lambda x: x**2 - 2, x=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fdiff import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
