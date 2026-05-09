"""Test freq_response_at (ztfrq)."""
import numpy as np
from moirais.fn.ztfrq import freq_response_at, ztfrq
from moirais.fn._containers import DescriptiveResult


class TestFreqResponseAt:
    def test_basic(self):
        b = [1.0, 0.5]
        a = [1.0, -0.8]
        result = freq_response_at(b, a, freqs=[0.0, 50.0, 100.0], fs=500.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "freq_response_at"

    def test_dc_gain(self):
        b = [1.0]
        a = [1.0]
        result = freq_response_at(b, a, freqs=[0.0], fs=100.0)
        assert abs(result.extra["magnitude"][0] - 1.0) < 1e-10

    def test_alias(self):
        assert ztfrq is freq_response_at
