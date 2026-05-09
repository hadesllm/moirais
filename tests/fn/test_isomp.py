"""Tests for moirais.fn.isomp."""
import numpy as np
from moirais.fn.isomp import isomap


def test_isomp_smoke():
    rng = np.random.default_rng(42)
    result = isomap(X=rng.standard_normal((30, 3)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.isomp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
