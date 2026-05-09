"""Tests for moirais.fn.cedfm."""
import numpy as np
from moirais.fn.cedfm import ecdf


def test_cedfm_smoke():
    rng = np.random.default_rng(42)
    result = ecdf(x=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cedfm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
