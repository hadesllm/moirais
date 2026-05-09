"""Tests for moirais.fn.eegbf -- EEG band filter."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from moirais.fn.eegbf import eegbf


class TestEegBf:
    def test_alpha_filter(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2560)
        result = eegbf(x, fs=256, band="alpha")
        assert result.name == "eeg_band_filter"
        assert result.filtered is not None
        assert len(result.filtered) == 2560
        assert result.extra["band"] == "alpha"

    def test_all_bands(self):
        x = np.random.default_rng(42).standard_normal(2560)
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            result = eegbf(x, fs=256, band=band)
            assert result.n_samples == 2560
            assert result.extra["band"] == band

    def test_unknown_band(self):
        x = np.random.default_rng(42).standard_normal(1024)
        result = eegbf(x, fs=256, band="invalid")
        assert np.array_equal(result.filtered, x)
