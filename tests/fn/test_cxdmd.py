"""Test complex_demodulation."""
import numpy as np
from moirais.fn.cxdmd import complex_demodulation, cxdmd
from moirais.fn._containers import DescriptiveResult


class TestComplexDemodulation:
    def test_basic(self):
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        result = complex_demodulation(x, fc=50.0, fs=fs)
        assert isinstance(result, DescriptiveResult)

    def test_envelope_length(self):
        fs = 500.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 20 * t)
        result = complex_demodulation(x, fc=20.0, fs=fs)
        assert len(result.extra["envelope"]) == len(t)

    def test_phase_length(self):
        fs = 500.0
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * 20 * t)
        result = complex_demodulation(x, fc=20.0, fs=fs)
        assert len(result.extra["phase"]) == len(t)

    def test_name(self):
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = complex_demodulation(x, fc=5.0, fs=100.0)
        assert result.name == "complex_demodulation"

    def test_alias(self):
        assert cxdmd is complex_demodulation
