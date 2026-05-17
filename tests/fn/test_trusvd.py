"""Tests for morie.fn.trusvd -- truncated SVD decomposition."""

import numpy as np
from morie.fn.trusvd import svd_rank_reduce, trusvd
from morie.fn._containers import DescriptiveResult


class TestTrusvd:
    def test_alias(self):
        assert trusvd is svd_rank_reduce

    def test_rank_reduction(self):
        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (10, 5))
        r = svd_rank_reduce(A, rank=2)
        assert isinstance(r, DescriptiveResult)
        assert r.value["rank"] == 2
        assert r.value["energy_retained"] < 1.0

    def test_full_rank(self):
        A = np.eye(5)
        r = svd_rank_reduce(A)
        assert r.value["reconstruction_error"] < 1e-8
        assert r.value["energy_retained"] > 0.999
