"""Test moving_average (mvavg)."""
import numpy as np
from moirais.fn.mvavg import moving_average, mvavg
from moirais.fn._containers import DescriptiveResult


class TestMovingAverage:
    def test_length_preserved(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(x, window=3)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 5

    def test_smoothing(self):
        x = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        result = moving_average(x, window=3)
        assert result.value[2] < 10.0

    def test_alias(self):
        assert mvavg is moving_average
