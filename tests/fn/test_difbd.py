"""Tests for difbd -- DIF bundle analysis."""
import numpy as np
from moirais.fn.difbd import dif_bundle
from moirais.fn._containers import DIFResult


class TestDifBundle:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (200, 10))
        group = np.array([0] * 100 + [1] * 100)
        bundles = {"bundle_1": [0, 1, 2], "bundle_2": [3, 4, 5]}
        result = dif_bundle(X, group, bundles)
        assert isinstance(result, DIFResult)
        assert result.method == "Bundle"

    def test_bundle_names_in_items(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (100, 6))
        group = np.array([0] * 50 + [1] * 50)
        bundles = {"A": [0, 1, 2], "B": [3, 4, 5]}
        result = dif_bundle(X, group, bundles)
        assert "bundle" in result.items.columns
