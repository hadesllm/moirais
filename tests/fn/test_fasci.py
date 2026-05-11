"""Test fastica (fasci)."""
import numpy as np
from morie.fn.fasci import fastica, fasci
from morie.fn._containers import DescriptiveResult


class TestFasci:
    def test_basic(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((200, 3))
        A = rng.standard_normal((3, 3))
        X = S @ A.T
        result = fastica(X, n_components=3, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "fastica"
        assert result.value == 3

    def test_sources_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        r = fastica(X, n_components=2, seed=0)
        assert r.extra["sources"].shape == (100, 2)

    def test_alias(self):
        assert fasci is fastica
