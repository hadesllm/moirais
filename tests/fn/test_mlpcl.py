"""Test mlp_classify (mlpcl)."""
import numpy as np
from moirais.fn.mlpcl import mlp_classify, mlpcl
from moirais.fn._containers import DescriptiveResult


class TestMlpcl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = mlp_classify(X[:50], y[:50], X[50:], hidden=(16, 8), n_iter=100)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "mlp_classify"
        assert len(result.extra["predictions"]) == 10

    def test_probabilities(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = mlp_classify(X[:50], y[:50], X[50:], hidden=(16,), n_iter=50)
        probs = result.extra["probabilities"]
        assert np.all((probs >= 0) & (probs <= 1))

    def test_alias(self):
        assert mlpcl is mlp_classify
