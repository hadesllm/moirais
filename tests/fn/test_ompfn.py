"""Test omp_sparse (ompfn)."""
import numpy as np
from moirais.fn.ompfn import omp_sparse, ompfn
from moirais.fn._containers import DescriptiveResult


class TestOmpfn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        D = rng.standard_normal((50, 100))
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        x = D[:, 3] * 2.0 + D[:, 17] * 1.5
        result = omp_sparse(D, x, sparsity=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "omp_sparse"
        assert result.value < 1.0

    def test_support_found(self):
        rng = np.random.default_rng(0)
        D = rng.standard_normal((30, 60))
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        x = D[:, 5] * 3.0
        r = omp_sparse(D, x, sparsity=3)
        assert 5 in r.extra["support"]

    def test_alias(self):
        assert ompfn is omp_sparse
