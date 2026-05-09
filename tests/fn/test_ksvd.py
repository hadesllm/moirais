"""Test ksvd_dictionary (ksvd)."""
import numpy as np
from moirais.fn.ksvd import ksvd_dictionary, ksvd
from moirais.fn._containers import DescriptiveResult


class TestKsvd:
    def test_basic(self):
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((10, 30))
        result = ksvd_dictionary(Y, n_atoms=8, sparsity=2, n_iter=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ksvd_dictionary"
        assert result.value >= 0

    def test_dictionary_shape(self):
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((8, 20))
        r = ksvd_dictionary(Y, n_atoms=12, sparsity=2, n_iter=3)
        assert r.extra["dictionary"].shape == (8, 12)
        assert r.extra["coefficients"].shape == (12, 20)

    def test_error_decreases(self):
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((6, 15))
        r1 = ksvd_dictionary(Y, n_atoms=10, sparsity=2, n_iter=1)
        r2 = ksvd_dictionary(Y, n_atoms=10, sparsity=2, n_iter=20)
        assert r2.value <= r1.value + 0.1

    def test_alias(self):
        assert ksvd is ksvd_dictionary
