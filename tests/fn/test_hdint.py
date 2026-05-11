"""Tests for morie.fn.hdint."""
import numpy as np
from morie.fn.hdint import hdint


def test_hdint_smoke():
    rng = np.random.default_rng(42)
    result = hdint(samples=rng.standard_normal(1000))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.hdint import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
