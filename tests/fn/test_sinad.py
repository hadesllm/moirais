"""Test sinad_compute (sinad)."""
import numpy as np
from moirais.fn.sinad import sinad_compute, sinad
from moirais.fn._containers import DescriptiveResult


class TestSinad:
    def test_basic(self):
        t = np.arange(256) / 256.0
        x = np.sin(2 * np.pi * 10 * t)
        result = sinad_compute(x, fs=256.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "sinad"
        assert result.value > 0

    def test_pure_tone_high(self):
        t = np.arange(1024) / 1024.0
        x = np.sin(2 * np.pi * 50 * t)
        result = sinad_compute(x, fs=1024.0)
        assert result.value > 20.0

    def test_alias(self):
        assert sinad is sinad_compute
