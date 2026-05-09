"""Tests for moirais.fn.thndm -- Mach shock wave."""

from moirais.fn.thndm import mach_shock, thndm
from moirais.fn._containers import DescriptiveResult
import pytest


class TestThndm:
    def test_alias(self):
        assert thndm is mach_shock

    def test_subsonic(self):
        r = mach_shock(100.0, speed_of_sound=343.0)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1.0
        assert r.extra["regime"] == "subsonic"

    def test_supersonic(self):
        r = mach_shock(686.0, speed_of_sound=343.0)
        assert r.value == pytest.approx(2.0, rel=0.01)
        assert r.extra["regime"] == "supersonic"
        assert r.extra["pressure_ratio"] > 1.0
