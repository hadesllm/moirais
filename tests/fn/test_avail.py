"""Tests for avail (system availability)."""
from moirais.fn.avail import availability


def test_availability_basic():
    r = availability(mtbf_val=1000, mttr=10)
    assert abs(r.value - 1000 / 1010) < 1e-10


def test_availability_nines():
    r = availability(mtbf_val=9999, mttr=1)
    assert r.extra["nines"] >= 3
