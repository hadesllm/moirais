"""Tests for morie.fn.mtocl — collision rate."""

import pytest
from morie.fn.mtocl import mto_collision_rate
from morie.fn._containers import CrimeResult


class TestCollisionRate:
    def test_basic(self):
        r = mto_collision_rate(500, 1e9)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(50.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            mto_collision_rate(10, 0)
