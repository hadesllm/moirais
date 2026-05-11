"""Test csp_filter (cspfn)."""
import numpy as np
from morie.fn.cspfn import csp_filter, cspfn
from morie.fn._containers import DescriptiveResult


class TestCspfn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((10, 4, 50))
        X2 = rng.standard_normal((10, 4, 50))
        result = csp_filter(X1, X2, n_components=2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "csp_filter"
        assert result.value >= 2

    def test_features_shape(self):
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((8, 6, 40))
        X2 = rng.standard_normal((8, 6, 40))
        r = csp_filter(X1, X2, n_components=2)
        assert r.extra["features_class1"].shape[0] == 8
        assert r.extra["features_class2"].shape[0] == 8

    def test_alias(self):
        assert cspfn is csp_filter
