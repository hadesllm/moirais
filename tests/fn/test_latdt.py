"""Test latency_detect (latdt)."""
import numpy as np
from morie.fn.latdt import latency_detect, latdt
from morie.fn._containers import DescriptiveResult


class TestLatencyDetect:
    def test_basic(self):
        signal = np.zeros(500)
        signal[110] = 5.0
        signal[310] = 5.0
        events = np.array([100, 300])
        result = latency_detect(signal, events, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "latency_detect"

    def test_correct_latency(self):
        signal = np.zeros(500)
        signal[120] = 10.0
        events = np.array([100])
        result = latency_detect(signal, events, fs=100.0)
        assert np.allclose(result.value, 0.2, atol=0.01)

    def test_multiple_events(self):
        signal = np.zeros(500)
        signal[110] = 5.0
        signal[310] = 5.0
        events = np.array([100, 300])
        result = latency_detect(signal, events, fs=100.0)
        assert result.extra["n_events"] == 2

    def test_alias(self):
        assert latdt is latency_detect
