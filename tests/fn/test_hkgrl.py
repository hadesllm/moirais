"""Tests for moirais.fn.hkgrl -- Hawking radiation temperature."""

from moirais.fn.hkgrl import hawking_temperature, hkgrl
from moirais.fn._containers import DescriptiveResult


class TestHkgrl:
    def test_alias(self):
        assert hkgrl is hawking_temperature

    def test_solar_mass(self):
        result = hawking_temperature(1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 1e-6

    def test_smaller_hotter(self):
        r1 = hawking_temperature(1.0)
        r2 = hawking_temperature(0.1)
        assert r2.value > r1.value
