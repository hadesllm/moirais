"""Test semi_supervised (sslcl)."""
import numpy as np
from moirais.fn.sslcl import semi_supervised, sslcl
from moirais.fn._containers import DescriptiveResult


class TestSslcl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X_l = np.vstack([rng.standard_normal((10, 2)) + 3, rng.standard_normal((10, 2)) - 3])
        y_l = np.array([1] * 10 + [0] * 10)
        X_u = rng.standard_normal((20, 2))
        result = semi_supervised(X_l, y_l, X_u, n_iter=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "semi_supervised"
        assert len(result.extra["predictions"]) == 20

    def test_alias(self):
        assert sslcl is semi_supervised
