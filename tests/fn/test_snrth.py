"""Test snr_threshold (snrth)."""
import numpy as np

from morie.fn.snrth import snr_threshold, snrth
from morie.fn._containers import DescriptiveResult


class TestSnrThreshold:
    def test_basic(self):
        result = snr_threshold(10.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "snr_threshold"

    def test_high_snr_meets(self):
        result = snr_threshold(20.0, target_ber=1e-3)
        assert result.extra["meets_threshold"] is True

    def test_low_snr_fails(self):
        result = snr_threshold(0.0, target_ber=1e-6)
        assert result.extra["meets_threshold"] is False

    def test_alias(self):
        assert snrth is snr_threshold
