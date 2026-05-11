"""Tests for morie.fn.pid."""
import numpy as np
from morie.fn.pid import pid


def test_pid_smoke():
    rng = np.random.default_rng(42)
    result = pid()
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.pid import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
