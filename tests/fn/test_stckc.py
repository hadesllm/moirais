"""Test stacking_classify (stckc)."""
import numpy as np
from morie.fn.stckc import stacking_classify, stckc
from morie.fn._containers import DescriptiveResult


class TestStckc:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = stacking_classify(X[:50], y[:50], X[50:])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "stacking_classify"
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert stckc is stacking_classify
