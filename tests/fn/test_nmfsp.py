"""Test nmf_sparse (nmfsp)."""
import numpy as np
from moirais.fn.nmfsp import nmf_sparse, nmfsp
from moirais.fn._containers import DescriptiveResult


class TestNmfsp:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.abs(rng.standard_normal((20, 15)))
        result = nmf_sparse(X, n_components=3, max_iter=50, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "nmf_sparse"
        assert result.value >= 0.0

    def test_nonneg(self):
        rng = np.random.default_rng(7)
        X = np.abs(rng.standard_normal((10, 8)))
        r = nmf_sparse(X, n_components=2, max_iter=50, seed=7)
        assert np.all(r.extra["W"] >= 0)
        assert np.all(r.extra["H"] >= 0)

    def test_alias(self):
        assert nmfsp is nmf_sparse
