"""Tests for morie.fn.bumbl -- spectral centroid."""

import numpy as np
from morie.fn.bumbl import spectral_centroid, bumbl
from morie.fn._containers import DescriptiveResult


class TestBumbl:
    def test_alias(self):
        assert bumbl is spectral_centroid

    def test_pure_tone(self):
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 100 * t)
        r = spectral_centroid(x, fs=fs)
        assert isinstance(r, DescriptiveResult)
        assert 90 < r.value < 110

    def test_dc_signal(self):
        x = np.ones(100)
        r = spectral_centroid(x, fs=100.0)
        assert r.value < 1.0
