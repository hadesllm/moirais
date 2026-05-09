"""Tests for moirais.fn.eegbp -- EEG band power."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from moirais.fn.eegbp import eegbp


class TestEegBp:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2560)
        result = eegbp(x, fs=256)
        assert result.name == "eeg_band_power"
        assert result.value > 0
        assert "delta" in result.extra
        assert "alpha" in result.extra
        assert "total_power" in result.extra

    def test_relative_powers_sum(self):
        x = np.random.default_rng(42).standard_normal(5120)
        result = eegbp(x, fs=256)
        rel_sum = sum(result.extra[f"{b}_relative"] for b in ["delta", "theta", "alpha", "beta", "gamma"])
        assert abs(rel_sum - 1.0) < 0.01
