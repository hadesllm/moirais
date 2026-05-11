"""Tests for morie.fn.gofmd -- MDS goodness of fit."""

from morie.fn.gofmd import gof_mds_interpret, gofmd


def test_gofmd_excellent():
    r = gofmd(0.01)
    assert r.name == "gof_mds_interpret"
    assert r.value == "excellent"


def test_gofmd_poor():
    r = gofmd(0.15)
    assert r.value == "poor"


def test_gofmd_unacceptable():
    r = gofmd(0.25)
    assert r.value == "unacceptable"


def test_gofmd_alias():
    assert gofmd is gof_mds_interpret
