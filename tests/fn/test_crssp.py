"""Tests for morie.fn.crssp."""
import numpy as np
from morie.fn.crssp import crssp


def test_crssp_smoke():
    rng = np.random.default_rng(42)
    result = crssp(table=np.array([[10, 20], [30, 40]]))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.crssp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
