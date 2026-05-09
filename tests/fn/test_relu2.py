"""Test relu_squared."""
import numpy as np
from moirais.fn.relu2 import relu_squared, relu2
from moirais.fn._containers import DescriptiveResult


class TestReluSquared:
    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = relu_squared(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "relu_squared"

    def test_values(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = relu_squared(x)
        expected = np.array([0.0, 0.0, 1.0, 4.0])
        assert np.allclose(result.extra["output"], expected)

    def test_sparsity(self):
        x = np.array([-1.0, -2.0, 0.0, 1.0])
        result = relu_squared(x)
        assert result.extra["sparsity"] == 0.75

    def test_alias(self):
        assert relu2 is relu_squared
