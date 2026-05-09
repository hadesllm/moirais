"""Test group_sparse_decompose (grpsp)."""
import numpy as np
from moirais.fn.grpsp import group_sparse_decompose, grpsp
from moirais.fn._containers import DescriptiveResult


class TestGrpsp:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(20)
        groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
        result = group_sparse_decompose(x, groups, lambda_=0.5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "group_sparse_decompose"

    def test_sparsity(self):
        x = np.zeros(12)
        x[:3] = [1.0, 2.0, 3.0]
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        r = group_sparse_decompose(x, groups, lambda_=0.5)
        assert r.value <= 4

    def test_high_lambda_zeros(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(8) * 0.01
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        r = group_sparse_decompose(x, groups, lambda_=10.0)
        assert r.value == 0

    def test_group_norms(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(6)
        groups = [[0, 1, 2], [3, 4, 5]]
        r = group_sparse_decompose(x, groups, lambda_=0.1)
        assert len(r.extra["group_norms"]) == 2

    def test_alias(self):
        assert grpsp is group_sparse_decompose
