"""Tests for scraw -- raw scoring."""
import numpy as np
from moirais.fn.scraw import raw_score
from moirais.fn._containers import DescriptiveResult


class TestRawScore:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (50, 5))
        result = raw_score(X)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["total_score"]) == 50

    def test_with_weights(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        result = raw_score(X, item_weights=[1, 2, 1])
        scores = result.value["total_score"]
        assert scores[0] == 1 * 1 + 2 * 2 + 3 * 1
