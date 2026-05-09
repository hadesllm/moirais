"""Test upsample (upsmp)."""
import numpy as np
from moirais.fn.upsmp import upsample, upsmp
from moirais.fn._containers import SignalResult


class TestUpsample:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = upsample(x, factor=3)
        assert isinstance(result, SignalResult)
        assert result.n_samples == 9

    def test_zeros_inserted(self):
        x = np.array([1.0, 2.0])
        result = upsample(x, factor=3)
        expected = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        np.testing.assert_array_equal(result.filtered, expected)

    def test_alias(self):
        assert upsmp is upsample
