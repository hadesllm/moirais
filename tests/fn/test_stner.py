"""Test stationarity_test (stner)."""
import numpy as np
import pytest

from moirais.fn.stner import stationarity_test, stner
from moirais.fn._containers import DescriptiveResult


class TestStationarityTest:
    def test_stationary_signal(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        result = stationarity_test(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "stationarity_test"
        assert result.extra["stationary"] is True

    def test_non_stationary(self):
        x = np.concatenate([np.zeros(250), np.ones(250) * 100])
        result = stationarity_test(x, n_segments=5)
        assert result.extra["stationary"] is False

    def test_too_short(self):
        with pytest.raises(ValueError):
            stationarity_test([1.0], n_segments=5)

    def test_alias(self):
        assert stner is stationarity_test
