"""Test time_freq_uncertainty (tfinq)."""
import numpy as np
import pytest

from moirais.fn.tfinq import time_freq_uncertainty, tfinq
from moirais.fn._containers import DescriptiveResult


class TestTimeFreqUncertainty:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        result = time_freq_uncertainty(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "time_freq_uncertainty"

    def test_above_bound(self):
        t = np.linspace(0, 1, 256)
        x = np.exp(-50 * (t - 0.5) ** 2) * np.cos(2 * np.pi * 10 * t)
        result = time_freq_uncertainty(x, fs=256.0)
        assert result.extra["above_bound"] is True

    def test_zero_energy(self):
        with pytest.raises(ValueError):
            time_freq_uncertainty(np.zeros(64))

    def test_alias(self):
        assert tfinq is time_freq_uncertainty
