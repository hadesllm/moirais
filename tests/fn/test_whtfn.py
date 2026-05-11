"""Test whitening (whtfn)."""
import numpy as np
from morie.fn.whtfn import whitening, whtfn
from morie.fn._containers import DescriptiveResult


class TestWhtfn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        result = whitening(X)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "whitening"

    def test_identity_covariance(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 3))
        r = whitening(X)
        Xw = r.extra["X_white"]
        cov = Xw.T @ Xw / (len(Xw) - 1)
        np.testing.assert_allclose(cov, np.eye(3), atol=0.15)

    def test_alias(self):
        assert whtfn is whitening
