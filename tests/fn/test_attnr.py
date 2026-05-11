"""Tests for attnr (attenuation correction)."""
from morie.fn.attnr import attenuation_ratio


def test_attenuation_ratio_basic():
    r = attenuation_ratio(r_obs=0.5, rel_x=0.8, rel_y=0.9)
    assert r.value > 0.5
    assert abs(r.value - 0.5 / (0.8 * 0.9) ** 0.5) < 1e-4


def test_cheatsheet():
    from morie.fn.attnr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
