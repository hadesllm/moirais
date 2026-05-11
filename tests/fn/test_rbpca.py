"""Tests for robust_pca_pp."""
import numpy as np, pytest
from morie.fn.rbpca import robust_pca_pp


class TestRobustPCA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        r = robust_pca_pp(X, n_components=2)
        assert r.measure == "robust_pca_pp"
        assert len(r.extra["robust_variances"]) == 2
        assert r.extra["robust_variances"][0] >= r.extra["robust_variances"][1]

    def test_too_few(self):
        with pytest.raises(ValueError):
            robust_pca_pp(np.array([[1, 2]]))
