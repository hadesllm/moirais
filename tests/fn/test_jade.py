"""Test jade_ica (jade)."""
import numpy as np
from moirais.fn.jade import jade_ica, jade
from moirais.fn._containers import DescriptiveResult


class TestJade:
    def test_basic(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((200, 3))
        A = rng.standard_normal((3, 3))
        X = S @ A.T
        result = jade_ica(X, n_components=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "jade_ica"
        assert result.value == 3

    def test_sources_shape(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((150, 2))
        r = jade_ica(X, n_components=2)
        assert r.extra["sources"].shape == (150, 2)

    def test_alias(self):
        assert jade is jade_ica
