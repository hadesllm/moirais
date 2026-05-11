"""Tests for morie.fn.sprbk."""
import numpy as np
from morie.fn.sprbk import spring_mass


def test_sprbk_smoke():
    rng = np.random.default_rng(42)
    result = spring_mass(m=1.0, k=1.0, x0=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.sprbk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
