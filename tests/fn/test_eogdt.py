"""Test eog_detect (eogdt)."""
import numpy as np
from moirais.fn.eogdt import eog_detect, eogdt
from moirais.fn._containers import DescriptiveResult


class TestEogDetect:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        result = eog_detect(x, fs=256.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "eog_detect"

    def test_detects_artifacts(self):
        x = np.zeros(500)
        x[100] = 100.0
        x[300] = -100.0
        result = eog_detect(x, fs=256.0)
        assert result.value >= 1

    def test_derivative_computed(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = eog_detect(x, fs=256.0)
        assert len(result.extra["derivative"]) == len(x) - 1

    def test_alias(self):
        assert eogdt is eog_detect
