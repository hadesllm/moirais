"""Tests for rirr — inter-rater reliability."""
import numpy as np
from morie.fn.rirr import rirr

def test_rirr_basic(rng):
    r1 = rng.integers(1, 6, 50)
    r2 = r1 + rng.choice([-1, 0, 0, 0, 1], 50)
    r2 = np.clip(r2, 1, 5)
    result = rirr(r1, r2)
    assert isinstance(result, dict)
    assert "kappa" in result


def test_cheatsheet():
    from morie.fn.rirr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
