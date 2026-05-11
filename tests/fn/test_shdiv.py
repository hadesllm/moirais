"""Tests for morie.fn.shdiv."""
import numpy as np
from morie.fn.shdiv import shdiv


def test_shdiv_smoke():
    abundances = np.array([20, 15, 10, 5, 3, 2, 1, 1])
    result = shdiv(abundances=abundances)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.shdiv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
