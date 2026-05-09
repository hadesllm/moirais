"""Tests for moirais.fn.plurl — Plurality voting."""
from moirais.fn.plurl import plurl


def test_plurl_basic():
    r = plurl([0, 0, 1, 2, 0])
    assert r.value == 0
    assert r.extra["counts"][0] == 3


def test_plurl_tie_resolved():
    r = plurl([0, 1, 0, 1, 2])
    assert r.value in (0, 1)


def test_plurl_pct():
    r = plurl([0, 0, 0, 1])
    assert abs(r.extra["winner_pct"] - 0.75) < 1e-10
