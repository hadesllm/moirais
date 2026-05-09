"""Tests for moirais.fn.polck — polarity check."""
from moirais.fn.polck import polck


def test_polck_no_flip():
    r = polck([1, 2, 3])
    assert r.name == "polarity_check"
    assert r.extra["flipped"] is False


def test_polck_flip():
    r = polck([3, 2, 1])
    assert r.extra["flipped"] is True
    assert r.extra["corrected"] == [-3, -2, -1]
