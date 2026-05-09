"""Tests for moirais.fn.lancs -- Lanczos algorithm."""

import numpy as np
from moirais.fn.lancs import lanczos, lancs
from moirais.fn._containers import DescriptiveResult


class TestLancs:
    def test_alias(self):
        assert lancs is lanczos

    def test_largest_eigenvalue(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        r = lanczos(A, k=5)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - 5.0) < 0.1

    def test_symmetric(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        r = lanczos(A, k=2)
        true_eigs = sorted(np.linalg.eigvalsh(A))[::-1]
        assert abs(r.value - true_eigs[0]) < 0.1
