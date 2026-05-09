"""Test prediction_gain (predg)."""
import numpy as np
from moirais.fn.predg import prediction_gain, predg
from moirais.fn._containers import DescriptiveResult


class TestPredg:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(100))
        result = prediction_gain(x, [0.9])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "prediction_gain"
        assert result.value > 0

    def test_alias(self):
        assert predg is prediction_gain
