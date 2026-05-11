"""Test bias_error (bserr)."""
import numpy as np

from morie.fn.bserr import bias_error, bserr
from morie.fn._containers import DescriptiveResult


class TestBiasError:
    def test_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = bias_error(y_true, y_pred)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bias_error"
        assert np.isclose(result.value, 0.5)

    def test_zero_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        result = bias_error(x, x)
        assert result.value == 0.0

    def test_negative_bias(self):
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([3.0, 3.0])
        result = bias_error(y_true, y_pred)
        assert result.value == -2.0

    def test_alias(self):
        assert bserr is bias_error
