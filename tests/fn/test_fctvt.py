"""Tests for morie.fn.fctvt — factor convert levels."""
from morie.fn.fctvt import fctvt


def test_fctvt_smoke():
    r = fctvt([3, 1, 2, 1, 3])
    assert r.name == "factor_convert_levels"
    assert r.value == 3
    assert r.extra["coded"] == [2, 0, 1, 0, 2]


def test_fctvt_custom_levels():
    r = fctvt([1, 2, 3], levels=[1, 2, 3, 4])
    assert r.value == 4
