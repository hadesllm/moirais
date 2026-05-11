"""Test fbm_synthesis."""
import numpy as np
from morie.fn.fbmsn import fbm_synthesis, fbmsn
from morie.fn._containers import SignalResult


class TestFBMSynthesis:
    def test_basic(self):
        result = fbm_synthesis(256)
        assert isinstance(result, SignalResult)

    def test_length(self):
        result = fbm_synthesis(512, H=0.7)
        assert len(result.filtered) == 512

    def test_n_samples(self):
        result = fbm_synthesis(128)
        assert result.n_samples == 128

    def test_name(self):
        result = fbm_synthesis(256)
        assert result.name == "fbm_synthesis"

    def test_alias(self):
        assert fbmsn is fbm_synthesis
