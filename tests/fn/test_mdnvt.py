"""Tests for moirais.fn.mdnvt — Median voter."""
import numpy as np
from moirais.fn.mdnvt import mdnvt


def test_mdnvt_odd():
    r = mdnvt([1, 3, 5, 7, 9])
    assert r.value == 5.0


def test_mdnvt_even():
    r = mdnvt([1, 3, 5, 7])
    assert r.value == 4.0


def test_mdnvt_weighted():
    r = mdnvt([1, 2, 3], weights=[1, 1, 100])
    assert r.value == 3.0


def test_mdnvt_extra():
    r = mdnvt([10, 20, 30])
    assert r.extra["n_voters"] == 3
