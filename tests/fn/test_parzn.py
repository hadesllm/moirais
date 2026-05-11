"""Test parzen_classify (parzn)."""
import numpy as np
from morie.fn.parzn import parzen_classify, parzn
from morie.fn._containers import DescriptiveResult


class TestParzn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((20, 2)) + 3, rng.standard_normal((20, 2)) - 3])
        y = np.array([1] * 20 + [0] * 20)
        result = parzen_classify(X[:30], y[:30], X[30:], h=1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "parzen_classify"
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert parzn is parzen_classify
