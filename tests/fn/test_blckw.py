"""Tests for morie.fn.blckw -- SNR degradation."""

import numpy as np
from morie.fn.blckw import snr_degradation, blckw
from morie.fn._containers import DescriptiveResult


class TestBlckw:
    def test_alias(self):
        assert blckw is snr_degradation

    def test_high_snr(self):
        x = np.sin(np.linspace(0, 10, 1000))
        r = snr_degradation(x, snr_db=40.0)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["actual_snr_db"] > 30

    def test_noise_types(self):
        x = np.ones(100)
        for nt in ["gaussian", "uniform", "impulsive"]:
            r = snr_degradation(x, snr_db=10.0, noise_type=nt)
            assert len(r.value) == 100
