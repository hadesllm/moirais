"""Test threshold_detect."""
import numpy as np
from moirais.fn.thrdt import threshold_detect, alias
from moirais.fn._containers import DescriptiveResult


class TestThresholdDetect:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = threshold_detect(x, threshold=0.5)
        assert isinstance(result, DescriptiveResult)

    def test_value(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = threshold_detect(x, threshold=0.5)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_indices_in_extra(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = threshold_detect(x, threshold=0.5)
        assert "indices" in result.extra

    def test_direction_below(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = threshold_detect(x, threshold=-0.5, direction="below")
        assert isinstance(result, DescriptiveResult)

    def test_alias(self):
        assert alias is threshold_detect
