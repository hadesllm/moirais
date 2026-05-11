"""Tests for morie.fn.ksmth."""
import numpy as np
from morie.fn.ksmth import ksmth


def test_ksmth_smoke():
    rng = np.random.default_rng(42)
    result = ksmth(x=rng.uniform(40, 45, size=20), y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ksmth import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
