"""Tests for moirais.fn.draxm -- truncated SVD decomposition."""

import numpy as np
from moirais.fn.draxm import destroyer_decompose, draxm
from moirais.fn._containers import DescriptiveResult


class TestDraxm:
    def test_alias(self):
        assert draxm is destroyer_decompose

    def test_rank_reduction(self):
        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (10, 5))
        r = destroyer_decompose(A, rank=2)
        assert isinstance(r, DescriptiveResult)
        assert r.value["rank"] == 2
        assert r.value["energy_retained"] < 1.0

    def test_full_rank(self):
        A = np.eye(5)
        r = destroyer_decompose(A)
        assert r.value["reconstruction_error"] < 1e-8
        assert r.value["energy_retained"] > 0.999
