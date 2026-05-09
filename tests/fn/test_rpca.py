"""Test robust_pca (rpca)."""
import numpy as np
from moirais.fn.rpca import robust_pca, rpca
from moirais.fn._containers import DescriptiveResult


class TestRpca:
    def test_basic(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((20, 2)) @ rng.standard_normal((2, 30))
        S = np.zeros((20, 30))
        S[3, 5] = 10.0
        S[10, 20] = -8.0
        X = L + S
        result = robust_pca(X)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "robust_pca"
        assert result.value >= 1

    def test_decomposition(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((15, 15))
        r = robust_pca(X, max_iter=50)
        recon = r.extra["L"] + r.extra["S"]
        np.testing.assert_allclose(recon, X, atol=0.5)

    def test_alias(self):
        assert rpca is robust_pca
