"""Test ewma_detect (ewmdt)."""
import numpy as np
from morie.fn.ewmdt import ewma_detect, ewmdt
from morie.fn._containers import DescriptiveResult


class TestEwmaDetect:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        result = ewma_detect(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ewma_detect"

    def test_detects_outlier(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        x[100] = 20.0
        result = ewma_detect(x, lambda_=0.2, L=3.0)
        assert result.value > 0

    def test_ewma_array(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = ewma_detect(x)
        assert len(result.extra["ewma"]) == len(x)

    def test_alias(self):
        assert ewmdt is ewma_detect
