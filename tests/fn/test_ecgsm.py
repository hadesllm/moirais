"""Tests for morie.fn.ecgsm -- simulated 12-lead ECG."""

import numpy as np

from morie.fn.ecgsm import ecgsm


class TestEcgSm:
    def test_basic(self):
        result = ecgsm(hr=72, duration=5.0, fs=500)
        assert result.name == "ecg_12lead_simulate"
        assert result.filtered is not None
        assert result.filtered.shape == (12, 2500)
        assert result.n_samples == 2500
        assert result.extra["n_leads"] == 12
        assert result.extra["hr"] == 72

    def test_different_hr(self):
        result = ecgsm(hr=120, duration=3.0, fs=250)
        assert result.filtered.shape[0] == 12
        assert result.filtered.shape[1] == 750
        assert result.extra["hr"] == 120

    def test_signal_not_flat(self):
        result = ecgsm(hr=60, duration=2.0, fs=500)
        assert np.std(result.filtered[0]) > 0
