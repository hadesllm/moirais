"""Tests for moirais.fn.tpsuof — use of force."""

import pytest
from moirais.fn.tpsuof import tps_use_of_force
from moirais.fn._containers import CrimeResult


class TestUseOfForce:
    def test_basic(self):
        types = ["CEW", "CEW", "Physical", "Firearm", "Physical"]
        r = tps_use_of_force(types, 1000)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.005)

    def test_type_counts(self):
        types = ["CEW"] * 3 + ["Physical"] * 7
        r = tps_use_of_force(types, 500)
        assert r.extra["type_counts"]["CEW"] == 3
        assert r.extra["type_counts"]["Physical"] == 7

    def test_invalid(self):
        with pytest.raises(ValueError):
            tps_use_of_force([], 0)
