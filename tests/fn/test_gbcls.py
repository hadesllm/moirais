"""Test gbm_classify_bio (gbcls)."""
import numpy as np
from morie.fn.gbcls import gbm_classify_bio, gbcls
from morie.fn._containers import DescriptiveResult


class TestGbcls:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = gbm_classify_bio(X[:50], y[:50], X[50:], n_estimators=20, max_depth=2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "gbm_classify_bio"
        assert "probabilities" in result.extra

    def test_alias(self):
        assert gbcls is gbm_classify_bio
