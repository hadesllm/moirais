"""Test energy_density (enrgy)."""
import numpy as np
from moirais.fn.enrgy import energy_density, enrgy
from moirais.fn._containers import DescriptiveResult


class TestEnrgy:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = energy_density(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "energy_density"
        assert result.value > 0

    def test_has_esd(self):
        x = np.ones(32)
        result = energy_density(x, fs=2.0)
        assert "esd" in result.extra
        assert "freqs" in result.extra

    def test_alias(self):
        assert enrgy is energy_density
