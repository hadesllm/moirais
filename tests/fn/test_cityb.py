"""Tests for moirais.fn.cityb — city block distance."""
from moirais.fn.cityb import cityb


def test_cityb_smoke():
    r = cityb([0, 0], [3, 4])
    assert r.name == "city_block_distance"
    assert r.value == 7.0


def test_cheatsheet():
    from moirais.fn.cityb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
