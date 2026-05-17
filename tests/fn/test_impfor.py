"""Tests for morie.fn.impfor -- impact force modeling."""

from morie.fn.impfor import impact_force, impfor
from morie.fn._containers import DescriptiveResult
import pytest


class TestImpfor:
    def test_alias(self):
        assert impfor is impact_force

    def test_impulse_method(self):
        r = impact_force(10.0, 5.0, duration=0.01)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(5000.0, rel=0.01)

    def test_energy_method(self):
        r = impact_force(10.0, 5.0, deformation=0.05)
        assert r.value == pytest.approx(2500.0, rel=0.01)
        assert r.extra["method"] == "energy-deformation"
