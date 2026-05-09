"""Tests for moirais.fn.ertdm."""
import numpy as np
from moirais.fn.ertdm import ertdm


def test_ertdm_smoke():
    rng = np.random.default_rng(42)
    result = ertdm(
        p=np.abs(rng.standard_normal(10)) + 0.01,
        q=np.abs(rng.standard_normal(10)) + 0.01
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ertdm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
