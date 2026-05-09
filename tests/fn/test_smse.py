"""Test mean_squared_error (smse)."""
import numpy as np
from moirais.fn.smse import mean_squared_error, smse
from moirais.fn._containers import DescriptiveResult


class TestMSE:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        result = mean_squared_error(x, x)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 0.0

    def test_known_error(self):
        x = np.array([1.0, 2.0, 3.0])
        x_hat = np.array([2.0, 3.0, 4.0])
        result = mean_squared_error(x, x_hat)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert smse is mean_squared_error
