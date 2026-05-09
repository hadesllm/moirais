"""Test swiglu."""
import numpy as np
from moirais.fn.swglu import swiglu, swglu
from moirais.fn._containers import DescriptiveResult


class TestSwiglu:
    def test_basic(self):
        x = np.array([1.0, -1.0, 0.5])
        result = swiglu(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "swiglu"

    def test_output_shape(self):
        x = np.array([1.0, 2.0, 3.0])
        result = swiglu(x)
        assert result.extra["output"].shape == x.shape

    def test_with_weights(self):
        x = np.array([[1.0, 2.0]])
        W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        W3 = np.array([[1.0, 0.0], [0.0, 1.0]])
        W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = swiglu(x, W1=W1, W2=W2, W3=W3)
        assert result.extra["output"].shape == (1, 2)

    def test_alias(self):
        assert swglu is swiglu
