"""Test gradient_stats."""
import numpy as np
from moirais.fn.grdst import gradient_stats, grdst
from moirais.fn._containers import DescriptiveResult


class TestGradientStats:
    def test_basic(self):
        grads = [np.array([1.0, -2.0, 3.0])]
        result = gradient_stats(grads)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "gradient_stats"

    def test_norm(self):
        grads = [np.array([3.0, 4.0])]
        result = gradient_stats(grads)
        assert abs(result.value - 5.0) < 1e-10

    def test_sparsity(self):
        grads = [np.array([0.0, 0.0, 1.0])]
        result = gradient_stats(grads)
        assert result.extra["sparsity"] > 0.5

    def test_alias(self):
        assert grdst is gradient_stats
