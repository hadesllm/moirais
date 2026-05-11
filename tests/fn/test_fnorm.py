"""Test feature_normalize (fnorm)."""
import numpy as np
from morie.fn.fnorm import feature_normalize, fnorm
from morie.fn._containers import DescriptiveResult


class TestFnorm:
    def test_zscore(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3)) * 10 + 5
        result = feature_normalize(X, method="zscore")
        assert isinstance(result, DescriptiveResult)
        Xn = result.extra["X_normalized"]
        assert np.allclose(Xn.mean(axis=0), 0, atol=0.1)

    def test_minmax(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        result = feature_normalize(X, method="minmax")
        Xn = result.extra["X_normalized"]
        assert Xn.min() >= -0.01
        assert Xn.max() <= 1.01

    def test_alias(self):
        assert fnorm is feature_normalize
