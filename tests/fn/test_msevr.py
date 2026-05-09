"""Test mse_variance_bias (msevr)."""
import numpy as np

from moirais.fn.msevr import mse_variance_bias, msevr
from moirais.fn._containers import DescriptiveResult


class TestMseVarianceBias:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = y_true + 0.1
        result = mse_variance_bias(y_true, y_pred)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "mse_variance_bias"

    def test_decomposition(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = mse_variance_bias(y_true, y_pred)
        assert np.isclose(result.extra["bias"], 0.1, atol=1e-10)
        assert np.isclose(result.value, result.extra["bias_sq"] + result.extra["variance"], atol=1e-10)

    def test_alias(self):
        assert msevr is mse_variance_bias
