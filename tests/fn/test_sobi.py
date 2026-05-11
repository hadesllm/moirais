"""Test sobi_bss (sobi)."""
import numpy as np
from morie.fn.sobi import sobi_bss, sobi
from morie.fn._containers import DescriptiveResult


class TestSobi:
    def test_basic(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((300, 3))
        A = rng.standard_normal((3, 3))
        X = S @ A.T
        result = sobi_bss(X, n_sources=3, n_lags=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "sobi_bss"
        assert result.value == 3

    def test_sources_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        r = sobi_bss(X, n_sources=4)
        assert r.extra["sources"].shape == (200, 4)

    def test_alias(self):
        assert sobi is sobi_bss
