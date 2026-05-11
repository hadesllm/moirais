"""Tests for morie.fn.psovl."""
import numpy as np
from morie.fn.psovl import psovl


def test_psovl_smoke():
    rng = np.random.default_rng(42)
    result = psovl(
        ps_treated=rng.integers(0, 2, size=20).astype(float),
        ps_control=rng.uniform(0.1, 0.9, size=30)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.psovl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
