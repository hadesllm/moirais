"""Test viras."""
import pytest
from moirais.fn.viras import virasoro_algebra


def test_viras_basic():
    r = virasoro_algebra(c=26.0, m=2, n=-2)
    assert r.value == 4
    assert r.extra["anomaly"] == pytest.approx(26.0 / 12.0 * 2 * 3)


def test_viras_no_anomaly():
    r = virasoro_algebra(c=26.0, m=2, n=1)
    assert r.extra["anomaly"] == 0.0
