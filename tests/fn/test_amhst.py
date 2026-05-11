"""Test amplitude_hist."""
import numpy as np
from morie.fn.amhst import amplitude_hist, alias
from morie.fn._containers import DescriptiveResult


class TestAmplitudeHist:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = amplitude_hist(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = amplitude_hist(x)
        assert result.value is None

    def test_extra_not_empty(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = amplitude_hist(x)
        assert len(result.extra) > 0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = amplitude_hist(x)
        assert result.name == "amplitude_histogram"

    def test_alias(self):
        assert alias is amplitude_hist
