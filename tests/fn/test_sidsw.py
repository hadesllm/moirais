"""Tests for morie.fn.sidsw -- impact force modeling."""

from morie.fn.sidsw import impact_force, sidsw
from morie.fn._containers import DescriptiveResult
import pytest


class TestSidsw:
    def test_alias(self):
        assert sidsw is impact_force

    def test_impulse_method(self):
        r = impact_force(10.0, 5.0, duration=0.01)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(5000.0, rel=0.01)

    def test_energy_method(self):
        r = impact_force(10.0, 5.0, deformation=0.05)
        assert r.value == pytest.approx(2500.0, rel=0.01)
        assert r.extra["method"] == "energy-deformation"
