"""Tests for bldal (Bland-Altman)."""
import numpy as np
from morie.fn.bldal import bland_altman


def test_bland_altman_agreement():
    rng = np.random.default_rng(42)
    m1 = rng.standard_normal(50)
    m2 = m1 + rng.normal(0, 0.1, 50)
    r = bland_altman(m1, m2)
    assert abs(r.value) < 0.5
    assert r.extra["loa_lower"] < r.extra["loa_upper"]


def test_cheatsheet():
    from morie.fn.bldal import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
