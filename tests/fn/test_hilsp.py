"""Test hilbert_spectrum (hilsp)."""
import numpy as np
from moirais.fn.hilsp import hilbert_spectrum, hilsp
from moirais.fn._containers import DescriptiveResult


class TestHilsp:
    def test_basic(self):
        t = np.linspace(0, 1, 256)
        imf1 = np.sin(2 * np.pi * 10 * t)
        imf2 = np.sin(2 * np.pi * 30 * t)
        imfs = np.array([imf1, imf2])
        result = hilbert_spectrum(imfs, fs=256.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "hilbert_spectrum"

    def test_single_imf(self):
        t = np.linspace(0, 1, 128)
        imf = np.sin(2 * np.pi * 5 * t)
        r = hilbert_spectrum(imf, fs=128.0)
        assert r.extra["spectrum"] is not None
        assert len(r.extra["frequencies"]) > 0

    def test_alias(self):
        assert hilsp is hilbert_spectrum
