"""Test resample_signal (rsmpl)."""
import numpy as np
from morie.fn.rsmpl import resample_signal, rsmpl
from morie.fn._containers import SignalResult


class TestResampleSignal:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(100)
        result = resample_signal(x, up=2, down=1)
        assert isinstance(result, SignalResult)
        assert result.name == "resample_signal"

    def test_upsample(self):
        x = np.ones(100)
        result = resample_signal(x, up=3, down=1)
        assert result.n_samples == 300

    def test_alias(self):
        assert rsmpl is resample_signal
