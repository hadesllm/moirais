"""Tests for morie.fn.damatt -- vibranium damping."""

import numpy as np
from morie.fn.damatt import damped_attenuation, damatt
from morie.fn._containers import DescriptiveResult


class TestDamatt:
    def test_alias(self):
        assert damatt is damped_attenuation

    def test_attenuates_signal(self):
        t = np.linspace(0, 1, 256)
        sig = np.sin(2 * np.pi * 10 * t)
        r = damped_attenuation(sig, fs=256, damping_ratio=0.5, natural_freq=10)
        assert isinstance(r, DescriptiveResult)
        assert r.value["output"].shape == sig.shape
        assert len(r.value["freqs"]) > 0

    def test_gain_shape(self):
        sig = np.ones(64)
        r = damped_attenuation(sig, fs=64)
        assert r.value["gain_db"].shape == r.value["freqs"].shape
