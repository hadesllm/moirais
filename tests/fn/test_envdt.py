"""Test envelope_detect (envdt)."""
import numpy as np
from moirais.fn.envdt import envelope_detect, envdt
from moirais.fn._containers import DescriptiveResult


class TestEnvelopeDetect:
    def test_basic(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 50 * t) * np.exp(-3 * t)
        result = envelope_detect(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "envelope_detect"

    def test_positive_envelope(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        result = envelope_detect(x)
        assert np.all(result.extra["envelope"] >= 0)

    def test_value_positive(self):
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = envelope_detect(x)
        assert result.value > 0

    def test_alias(self):
        assert envdt is envelope_detect
