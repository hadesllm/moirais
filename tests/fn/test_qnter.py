"""Test quantization_error (qnter)."""
import numpy as np
from moirais.fn.qnter import quantization_error, qnter
from moirais.fn._containers import DescriptiveResult


class TestQuantizationError:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(100)
        result = quantization_error(x, bits=8)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "quantization_error"

    def test_mse_decreases_with_bits(self):
        x = np.random.default_rng(42).standard_normal(1000)
        mse4 = quantization_error(x, bits=4).value
        mse8 = quantization_error(x, bits=8).value
        assert mse8 < mse4

    def test_alias(self):
        assert qnter is quantization_error
