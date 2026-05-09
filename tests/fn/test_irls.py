"""Test iteratively_reweighted_ls (irls)."""
import numpy as np
from moirais.fn.irls import iteratively_reweighted_ls, irls
from moirais.fn._containers import DescriptiveResult


class TestIrls:
    def test_basic(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 15))
        x_true = np.zeros(15)
        x_true[0] = 1.0
        x_true[5] = -2.0
        b = A @ x_true
        result = iteratively_reweighted_ls(A, b, p=1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "iteratively_reweighted_ls"
        assert result.value >= 0

    def test_constraint_satisfied(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 10))
        b = rng.standard_normal(3)
        r = iteratively_reweighted_ls(A, b, p=1.0, n_iter=100)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-3)

    def test_p2_is_lstsq(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 8))
        b = rng.standard_normal(4)
        r = iteratively_reweighted_ls(A, b, p=2.0)
        x_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(r.extra["x"], x_ls, atol=1e-3)

    def test_alias(self):
        assert irls is iteratively_reweighted_ls
