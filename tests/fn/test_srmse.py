"""Test root_mean_squared_error (srmse)."""
import numpy as np
from morie.fn.srmse import root_mean_squared_error, srmse
from morie.fn._containers import DescriptiveResult


class TestRMSE:
    def test_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        result = root_mean_squared_error(x, x)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 0.0

    def test_known(self):
        x = np.array([1.0, 2.0, 3.0])
        x_hat = np.array([2.0, 3.0, 4.0])
        result = root_mean_squared_error(x, x_hat)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert srmse is root_mean_squared_error
