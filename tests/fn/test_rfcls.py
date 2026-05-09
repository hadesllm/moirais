"""Test rf_classify_bio (rfcls)."""
import numpy as np
from moirais.fn.rfcls import rf_classify_bio, rfcls
from moirais.fn._containers import DescriptiveResult


class TestRfcls:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = rf_classify_bio(X[:50], y[:50], X[50:], n_trees=10)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rf_classify_bio"
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert rfcls is rf_classify_bio
