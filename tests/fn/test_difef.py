"""Tests for difef — DIF effect size."""
import numpy as np
from moirais.fn.difef import difef

def test_difef_basic():
    mh_or = np.array([1.0, 1.5, 0.8, 2.0, 0.6])
    result = difef(mh_or, item_names=["i1","i2","i3","i4","i5"])
    assert len(result) == 5


def test_cheatsheet():
    from moirais.fn.difef import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
