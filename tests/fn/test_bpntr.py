"""Tests for moirais.fn.bpntr -- vibranium damping."""

import numpy as np
from moirais.fn.bpntr import vibranium_damping, bpntr
from moirais.fn._containers import DescriptiveResult


class TestBpntr:
    def test_alias(self):
        assert bpntr is vibranium_damping

    def test_attenuates_signal(self):
        t = np.linspace(0, 1, 256)
        sig = np.sin(2 * np.pi * 10 * t)
        r = vibranium_damping(sig, fs=256, damping_ratio=0.5, natural_freq=10)
        assert isinstance(r, DescriptiveResult)
        assert r.value["output"].shape == sig.shape
        assert len(r.value["freqs"]) > 0

    def test_gain_shape(self):
        sig = np.ones(64)
        r = vibranium_damping(sig, fs=64)
        assert r.value["gain_db"].shape == r.value["freqs"].shape
