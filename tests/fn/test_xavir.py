"""Tests for morie.fn.xavir."""
import numpy as np
from morie.fn.xavir import xavier_init


def test_xavir_smoke():
    rng = np.random.default_rng(42)
    result = xavier_init(fan_in=256, fan_out=128)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.xavir import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
