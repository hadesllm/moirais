"""Tests for moirais.fn.arnld -- Arnoldi iteration."""

import numpy as np
from moirais.fn.arnld import arnoldi, arnld
from moirais.fn._containers import DescriptiveResult


class TestArnld:
    def test_alias(self):
        assert arnld is arnoldi

    def test_hessenberg(self):
        A = np.random.default_rng(42).standard_normal((5, 5))
        r = arnoldi(A, k=4)
        assert isinstance(r, DescriptiveResult)
        H = r.extra["H"]
        for i in range(4):
            for j in range(i + 2, 4):
                assert abs(H[j, i]) < 1e-10

    def test_ritz_values(self):
        A = np.diag([1.0, 2.0, 3.0])
        r = arnoldi(A, k=3)
        assert r.value > 2.5
