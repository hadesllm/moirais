"""Tests for moirais.fn.optic -- OPTICS clustering."""

import numpy as np
from moirais.fn.optic import optics
from moirais.fn._containers import DescriptiveResult


class TestOptics:
    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        res = optics(X, min_samples=3)
        assert isinstance(res, DescriptiveResult)

    def test_labels_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        res = optics(X, min_samples=3)
        assert len(res.value) == 30

    def test_has_ordering(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        res = optics(X, min_samples=3)
        assert "ordering" in res.extra
        assert "reachability" in res.extra

    def test_finds_clusters(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(5, 0.3, (20, 2))])
        res = optics(X, min_samples=3)
        assert res.extra["n_clusters"] >= 1
