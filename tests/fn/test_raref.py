"""Tests for morie.fn.raref."""
import numpy as np
from morie.fn.raref import raref


def test_raref_smoke():
    rng = np.random.default_rng(42)
    result = raref(abundances=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.raref import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
