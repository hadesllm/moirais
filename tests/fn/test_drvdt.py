"""Test derivative_detect."""
import numpy as np
from morie.fn.drvdt import derivative_detect, alias
from morie.fn._containers import DescriptiveResult


class TestDerivativeDetect:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = derivative_detect(x)
        assert isinstance(result, DescriptiveResult)

    def test_value(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = derivative_detect(x)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_peaks_in_extra(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = derivative_detect(x)
        assert "peaks" in result.extra

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = derivative_detect(x)
        assert result.name == "derivative_detect"

    def test_alias(self):
        assert alias is derivative_detect
