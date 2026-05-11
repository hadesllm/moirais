"""Tests for morie.fn.shang -- circular harmonic analysis."""

import numpy as np
from morie.fn.shang import ring_harmonics, shang
from morie.fn._containers import DescriptiveResult


class TestShang:
    def test_alias(self):
        assert shang is ring_harmonics

    def test_pure_sine(self):
        N = 64
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        sig = 3.0 * np.sin(t)
        r = ring_harmonics(sig, n_harmonics=5)
        assert isinstance(r, DescriptiveResult)
        assert r.value["dominant_harmonic"] == 1
        assert r.value["amplitudes"][0] > r.value["amplitudes"][1]

    def test_dc_offset(self):
        sig = np.ones(32) * 5.0
        r = ring_harmonics(sig)
        assert abs(r.value["dc_offset"] - 5.0) < 0.01
