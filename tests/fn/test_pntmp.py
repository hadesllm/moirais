"""Test pan_tompkins_qrs (pntmp)."""
import numpy as np
import pytest

from moirais.fn.pntmp import pan_tompkins_qrs, pntmp
from moirais.fn._containers import SignalResult


class TestPanTompkinsQRS:
    def test_basic(self):
        rng = np.random.default_rng(42)
        t = np.arange(0, 5, 1/360)
        ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(len(t))
        result = pan_tompkins_qrs(ecg, fs=360.0)
        assert isinstance(result, SignalResult)
        assert result.name == "pan_tompkins_qrs"

    def test_extra_keys(self):
        t = np.arange(0, 3, 1/360)
        ecg = np.sin(2 * np.pi * 1.0 * t)
        result = pan_tompkins_qrs(ecg, fs=360.0)
        assert "qrs_indices" in result.extra
        assert "num_beats" in result.extra
        assert "heart_rate" in result.extra

    def test_fs_stored(self):
        ecg = np.random.default_rng(42).standard_normal(1000)
        result = pan_tompkins_qrs(ecg, fs=500.0)
        assert result.fs == 500.0

    def test_alias(self):
        assert pntmp is pan_tompkins_qrs
