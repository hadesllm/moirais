"""Test snr_improvement_fn (snri)."""
import numpy as np
import pytest

from moirais.fn.snri import snr_improvement_fn, snri
from moirais.fn._containers import DescriptiveResult


class TestSnrImprovement:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        noise = 0.5 * rng.standard_normal(256)
        result = snr_improvement_fn(
            x_noisy=x + noise,
            x_clean=x,
            x_filtered=x + 0.05 * noise,
        )
        assert isinstance(result, DescriptiveResult)
        assert result.name == "snr_improvement"

    def test_value_is_float(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        noise = 0.5 * rng.standard_normal(256)
        result = snr_improvement_fn(
            x_noisy=x + noise,
            x_clean=x,
            x_filtered=x + 0.05 * noise,
        )
        assert isinstance(result.value, float)

    def test_improvement_positive(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        noise = 0.5 * rng.standard_normal(256)
        result = snr_improvement_fn(
            x_noisy=x + noise,
            x_clean=x,
            x_filtered=x + 0.05 * noise,
        )
        assert result.value > 0

    def test_unit_in_extra(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        noise = 0.5 * rng.standard_normal(256)
        result = snr_improvement_fn(
            x_noisy=x + noise,
            x_clean=x,
            x_filtered=x + 0.05 * noise,
        )
        assert result.extra.get("unit") == "dB"

    def test_alias(self):
        assert snri is snr_improvement_fn
