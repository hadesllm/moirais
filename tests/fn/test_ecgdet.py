"""Tests for ecgdet — Pan-Tompkins QRS detector."""
import numpy as np
from morie.fn.ecgdet import pan_tompkins
from morie.fn._containers import SignalResult


def test_ecgdet_basic(ecg_synthetic):
    ecg, fs, _ = ecg_synthetic
    result = pan_tompkins(ecg, fs)
    assert isinstance(result, SignalResult)
    assert "r_peaks" in result.extra


def test_ecgdet_finds_peaks(ecg_synthetic):
    ecg, fs, true_peaks = ecg_synthetic
    result = pan_tompkins(ecg, fs)
    detected = result.extra["r_peaks"]
    assert len(detected) >= len(true_peaks) - 1
    for tp in true_peaks:
        diffs = np.abs(detected - tp)
        assert np.min(diffs) < 20
