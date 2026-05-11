"""Test snr_estimate_fn (snr)."""
import numpy as np
import pytest

from morie.fn.snr import snr_estimate_fn, snr
from morie.fn._containers import DescriptiveResult


class TestSnrEstimate:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        noise = 0.1 * np.random.default_rng(43).standard_normal(256)
        result = snr_estimate_fn(x, noise)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "snr_estimate"

    def test_value_is_float(self):
        x = np.random.default_rng(42).standard_normal(256)
        noise = 0.1 * np.random.default_rng(43).standard_normal(256)
        result = snr_estimate_fn(x, noise)
        assert isinstance(result.value, float)

    def test_high_snr(self):
        x = np.random.default_rng(42).standard_normal(256)
        noise = 0.001 * np.random.default_rng(43).standard_normal(256)
        result = snr_estimate_fn(x, noise)
        assert result.value > 0

    def test_unit_in_extra(self):
        x = np.random.default_rng(42).standard_normal(256)
        noise = 0.1 * np.random.default_rng(43).standard_normal(256)
        result = snr_estimate_fn(x, noise)
        assert result.extra.get("unit") == "dB"

    def test_alias(self):
        assert snr is snr_estimate_fn
