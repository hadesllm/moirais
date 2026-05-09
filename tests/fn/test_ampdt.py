"""Test amplitude_detect (ampdt)."""
import numpy as np
from moirais.fn.ampdt import amplitude_detect, ampdt
from moirais.fn._containers import DescriptiveResult


class TestAmplitudeDetect:
    def test_basic(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        peaks = np.array([25, 75, 125, 175])
        result = amplitude_detect(signal, peaks)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "amplitude_detect"

    def test_correct_amplitudes(self):
        signal = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        peaks = np.array([1, 3, 5])
        result = amplitude_detect(signal, peaks)
        assert np.allclose(result.extra["amplitudes"], [1.0, 2.0, 3.0])

    def test_invalid_peaks_filtered(self):
        signal = np.ones(10)
        peaks = np.array([-1, 5, 100])
        result = amplitude_detect(signal, peaks)
        assert result.extra["n_peaks"] == 1

    def test_alias(self):
        assert ampdt is amplitude_detect
