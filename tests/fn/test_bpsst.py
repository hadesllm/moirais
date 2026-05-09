"""Test bpsst."""
import pytest
from moirais.fn.bpsst import bps_state


def test_bpsst_basic():
    r = bps_state(charges=[3.0, 4.0])
    assert r.value == pytest.approx(5.0)
    assert r.extra["is_bps"]


def test_bpsst_not_bps():
    r = bps_state(charges=[3.0, 4.0], mass=10.0)
    assert not r.extra["is_bps"]
