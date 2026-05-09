"""Test bagging_classify (bgcls)."""
import numpy as np
from moirais.fn.bgcls import bagging_classify, bgcls
from moirais.fn._containers import DescriptiveResult


class TestBgcls:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + 2, rng.standard_normal((30, 2)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = bagging_classify(X[:50], y[:50], X[50:], n_estimators=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bagging_classify"
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert bgcls is bagging_classify
