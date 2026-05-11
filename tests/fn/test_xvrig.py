"""Test xavier_init."""
import numpy as np
from morie.fn.xvrig import xavier_init, xvrig
from morie.fn._containers import DescriptiveResult


class TestXavierInit:
    def test_basic(self):
        result = xavier_init(64, 128, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "xavier_init"

    def test_shape(self):
        result = xavier_init(64, 128, seed=42)
        assert result.extra["weights"].shape == (128, 64)

    def test_uniform(self):
        result = xavier_init(64, 128, distribution="uniform", seed=42)
        a = np.sqrt(6.0 / (64 + 128))
        w = result.extra["weights"]
        assert np.all(w >= -a) and np.all(w <= a)

    def test_alias(self):
        assert xvrig is xavier_init
