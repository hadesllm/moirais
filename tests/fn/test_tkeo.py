"""Test teager_energy (tkeo)."""
import numpy as np
from morie.fn.tkeo import teager_energy, tkeo
from morie.fn._containers import DescriptiveResult


class TestTeagerEnergy:
    def test_basic(self):
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 10 * t)
        result = teager_energy(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "teager_energy"

    def test_positive_energy(self):
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 10 * t)
        result = teager_energy(x)
        assert result.value > 0

    def test_short_signal(self):
        result = teager_energy(np.array([1.0, 2.0]))
        assert len(result.extra["energy"]) == 0

    def test_dc_zero_energy(self):
        x = np.ones(100)
        result = teager_energy(x)
        assert np.allclose(result.extra["energy"], 0.0)

    def test_alias(self):
        assert tkeo is teager_energy
