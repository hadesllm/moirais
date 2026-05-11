"""Test qrs_waveform_features."""
import numpy as np
from morie.fn.qrswv import qrs_waveform_features, qrswv
from morie.fn._containers import DescriptiveResult


class TestQRSWaveformFeatures:
    def test_basic(self):
        beat = np.array([0, 0.1, 0.5, 1.0, 0.3, -0.2, -0.1, 0])
        result = qrs_waveform_features(beat)
        assert isinstance(result, DescriptiveResult)

    def test_amplitude(self):
        beat = np.array([0, 0.1, 0.5, 1.0, 0.3, -0.2, -0.1, 0])
        result = qrs_waveform_features(beat)
        assert result.value == 1.0

    def test_extra_keys(self):
        beat = np.array([0, 0.1, 0.5, 1.0, 0.3, -0.2, -0.1, 0])
        result = qrs_waveform_features(beat)
        for key in ("amplitude", "duration", "area", "slope_up", "slope_down", "peak_index"):
            assert key in result.extra

    def test_name(self):
        beat = np.random.default_rng(42).standard_normal(50)
        result = qrs_waveform_features(beat)
        assert result.name == "qrs_waveform_features"

    def test_alias(self):
        assert qrswv is qrs_waveform_features
