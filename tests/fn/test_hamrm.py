"""Tests for morie.fn.hamrm -- impact energy."""

import pytest
from morie.fn.hamrm import impact_energy, hamrm
from morie.fn._containers import DescriptiveResult


class TestHamrm:
    def test_alias(self):
        assert hamrm is impact_energy

    def test_basic(self):
        result = impact_energy(1.5, 0.5, pendulum_mass=25.0)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 25.0 * 9.81 * 1.0) < 0.1

    def test_with_area(self):
        result = impact_energy(2.0, 1.0, specimen_area=0.01)
        assert "impact_strength_J_m2" in result.extra

    def test_invalid(self):
        with pytest.raises(ValueError):
            impact_energy(0.5, 1.5)
