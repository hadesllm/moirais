"""Tests for qrsdt -- QRS complex detection."""
import numpy as np
from moirais.fn.qrsdt import qrsdt
from moirais.fn._containers import DescriptiveResult


def test_qrsdt_basic():
    rng = np.random.default_rng(42)
    fs = 360
    t = np.arange(0, 5.0, 1 / fs)
    ecg = np.zeros_like(t)
    for beat_t in np.arange(0.5, 5.0, 0.8):
        idx = int(beat_t * fs)
        if idx < len(ecg):
            ecg[idx] = 1.0
    ecg += rng.standard_normal(len(t)) * 0.01
    result = qrsdt(ecg, fs)
    assert isinstance(result, DescriptiveResult)
    assert "r_peaks" in result.extra


def test_qrsdt_finds_peaks():
    fs = 500
    t = np.arange(0, 3.0, 1 / fs)
    ecg = np.zeros_like(t)
    beat_times = [0.5, 1.3, 2.1]
    for bt in beat_times:
        idx = int(bt * fs)
        lo = max(0, idx - 5)
        hi = min(len(ecg), idx + 5)
        ecg[lo:hi] = np.hanning(hi - lo) * 2.0
    result = qrsdt(ecg, fs)
    assert len(result.extra["r_peaks"]) >= 2


def test_qrsdt_rr_intervals():
    fs = 360
    t = np.arange(0, 5.0, 1 / fs)
    ecg = np.zeros_like(t)
    for bt in np.arange(0.5, 5.0, 1.0):
        idx = int(bt * fs)
        lo = max(0, idx - 3)
        hi = min(len(ecg), idx + 3)
        ecg[lo:hi] = 1.0
    result = qrsdt(ecg, fs)
    rr = result.extra["rr_intervals"]
    if len(rr) > 0:
        assert np.all(rr > 0)
