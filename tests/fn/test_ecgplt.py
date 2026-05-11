"""Tests for morie.fn.ecgplt -- ECG multi-lead plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.ecgplt import ecgplt


class TestEcgPlt:
    def test_basic_12lead(self):
        signals = np.random.default_rng(42).standard_normal((12, 1000))
        result = ecgplt(signals, fs=500)
        assert result.name == "ecg_plot"
        assert result.value == 12
        assert result.extra["figure"] is not None
        assert result.extra["n_leads"] == 12
        plt.close(result.extra["figure"])

    def test_single_lead(self):
        signal = np.random.default_rng(42).standard_normal(500)
        result = ecgplt(signal, fs=250)
        assert result.value == 1
        plt.close(result.extra["figure"])

    def test_with_duration(self):
        signals = np.random.default_rng(42).standard_normal((3, 2000))
        result = ecgplt(signals, fs=500, duration=2.0)
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])

    def test_with_lead_names(self):
        signals = np.random.default_rng(42).standard_normal((2, 500))
        result = ecgplt(signals, fs=500, lead_names=["I", "II"])
        assert result.value == 2
        plt.close(result.extra["figure"])
