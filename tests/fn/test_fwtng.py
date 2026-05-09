"""Test feature_whiten (fwtng)."""
import numpy as np
from moirais.fn.fwtng import feature_whiten, fwtng
from moirais.fn._containers import DescriptiveResult


class TestFwtng:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        result = feature_whiten(X)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "feature_whiten"
        assert result.extra["X_whitened"].shape == (50, 4)

    def test_covariance_identity(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        result = feature_whiten(X, method="zca")
        cov = np.cov(result.extra["X_whitened"].T)
        assert np.allclose(cov, np.eye(3), atol=0.1)

    def test_alias(self):
        assert fwtng is feature_whiten
