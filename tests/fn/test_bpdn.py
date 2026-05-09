"""Test basis_pursuit (bpdn)."""
import numpy as np
from moirais.fn.bpdn import basis_pursuit, bpdn
from moirais.fn._containers import DescriptiveResult


class TestBpdn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        D = rng.standard_normal((50, 100))
        x = D[:, 7] * 2.0 + D[:, 22] * 1.0
        result = basis_pursuit(D, x, lambda_=0.01)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "basis_pursuit"

    def test_sparsity(self):
        rng = np.random.default_rng(1)
        D = rng.standard_normal((30, 60))
        x = D[:, 10] * 5.0
        r = basis_pursuit(D, x, lambda_=0.5, max_iter=300)
        assert r.extra["n_nonzero"] < 30

    def test_alias(self):
        assert bpdn is basis_pursuit
