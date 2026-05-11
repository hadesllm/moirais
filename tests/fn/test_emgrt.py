"""Test emg_rms_threshold (emgrt)."""
import numpy as np
from morie.fn.emgrt import emg_rms_threshold, emgrt
from morie.fn._containers import DescriptiveResult


class TestEmgRmsThreshold:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(200) * 0.01,
                            rng.standard_normal(100) * 3.0,
                            rng.standard_normal(200) * 0.01])
        result = emg_rms_threshold(x, window=30)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "emg_rms_threshold"

    def test_detects_burst(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(200) * 0.01,
                            rng.standard_normal(100) * 5.0,
                            rng.standard_normal(200) * 0.01])
        result = emg_rms_threshold(x, window=30, threshold_factor=2.0)
        assert result.value >= 1

    def test_onsets_offsets(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(200) * 0.01,
                            rng.standard_normal(100) * 5.0,
                            rng.standard_normal(200) * 0.01])
        result = emg_rms_threshold(x, window=30)
        assert "onsets" in result.extra
        assert "offsets" in result.extra

    def test_alias(self):
        assert emgrt is emg_rms_threshold
