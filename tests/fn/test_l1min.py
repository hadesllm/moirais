"""Test l1_minimize (l1min)."""
import numpy as np
from moirais.fn.l1min import l1_minimize, l1min
from moirais.fn._containers import DescriptiveResult


class TestL1min:
    def test_basic(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 50))
        x_true = np.zeros(50)
        x_true[:5] = rng.standard_normal(5)
        b = A @ x_true
        result = l1_minimize(A, b, lambda_=0.01, max_iter=1000)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "l1_minimize"
        assert result.value >= 0

    def test_sparse_recovery(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 60))
        x_true = np.zeros(60)
        x_true[0] = 3.0
        x_true[10] = -2.0
        x_true[20] = 1.5
        b = A @ x_true
        r = l1_minimize(A, b, lambda_=0.001, max_iter=2000)
        assert r.extra["sparsity"] <= 60

    def test_residual(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 20))
        b = rng.standard_normal(10)
        r = l1_minimize(A, b, lambda_=0.05)
        expected_res = b - A @ r.extra["x"]
        np.testing.assert_allclose(r.extra["residual"], expected_res, atol=1e-10)

    def test_alias(self):
        assert l1min is l1_minimize
