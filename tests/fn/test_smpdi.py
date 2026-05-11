"""Tests for morie.fn.smpdi."""
import numpy as np
from morie.fn.smpdi import smpdi


def test_smpdi_smoke():
    abundances = np.array([20, 15, 10, 5, 3, 2, 1, 1])
    result = smpdi(abundances=abundances)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.smpdi import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
