"""Test amuse_bss (amuse)."""
import numpy as np
from morie.fn.amuse import amuse_bss, amuse
from morie.fn._containers import DescriptiveResult


class TestAmuse:
    def test_basic(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((200, 3))
        A = rng.standard_normal((3, 3))
        X = S @ A.T
        result = amuse_bss(X, lag=1)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "amuse_bss"
        assert result.value == 3

    def test_sources_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((150, 2))
        r = amuse_bss(X, lag=2)
        assert r.extra["sources"].shape == (150, 2)

    def test_alias(self):
        assert amuse is amuse_bss
