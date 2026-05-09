"""Test mean_absolute_error (smae)."""
import numpy as np
from moirais.fn.smae import mean_absolute_error, smae
from moirais.fn._containers import DescriptiveResult


class TestMAE:
    def test_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        result = mean_absolute_error(x, x)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 0.0

    def test_known(self):
        x = np.array([1.0, 2.0, 3.0])
        x_hat = np.array([2.0, 3.0, 4.0])
        assert abs(mean_absolute_error(x, x_hat).value - 1.0) < 1e-10

    def test_alias(self):
        assert smae is mean_absolute_error
