"""Tests for morie.fn.csrip."""
import numpy as np
from morie.fn.csrip import csrip


def test_csrip_smoke():
    rng = np.random.default_rng(42)
    result = csrip(A=rng.standard_normal((20, 30)), s=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.csrip import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
