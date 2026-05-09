"""Test learning_vq (lvq)."""
import numpy as np
from moirais.fn.lvq import learning_vq, lvq
from moirais.fn._containers import DescriptiveResult


class TestLvq:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = learning_vq(X[:50], y[:50], X[50:], n_proto=3, n_iter=20)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "learning_vq"
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert lvq is learning_vq
