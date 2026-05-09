"""Test hilbert_envelope (hilev)."""
import numpy as np
from moirais.fn.hilev import hilbert_envelope, hilev
from moirais.fn._containers import DescriptiveResult


class TestHilbertEnvelope:
    def test_basic(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 10 * t)
        result = hilbert_envelope(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "hilbert_envelope"

    def test_envelope_nonnegative(self):
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 5 * t)
        result = hilbert_envelope(x)
        assert np.all(result.extra["envelope"] >= 0)

    def test_sine_envelope_near_one(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 10 * t)
        result = hilbert_envelope(x)
        env = result.extra["envelope"]
        assert np.allclose(env[100:900], 1.0, atol=0.05)

    def test_alias(self):
        assert hilev is hilbert_envelope
