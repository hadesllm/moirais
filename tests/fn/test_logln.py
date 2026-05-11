"""Tests for morie.fn.logln."""
import numpy as np
from morie.fn.logln import logln


def test_logln_smoke():
    rng = np.random.default_rng(42)
    result = logln(table=np.array([[10, 20], [30, 40]]))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.logln import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
