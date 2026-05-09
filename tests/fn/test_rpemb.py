"""Test rotary_embed."""
import numpy as np
from moirais.fn.rpemb import rotary_embed, rpemb
from moirais.fn._containers import DescriptiveResult


class TestRotaryEmbed:
    def test_basic_2d(self):
        x = np.random.default_rng(42).standard_normal((4, 8))
        result = rotary_embed(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rotary_embed"
        assert result.extra["output"].shape == (4, 8)

    def test_basic_3d(self):
        x = np.random.default_rng(42).standard_normal((2, 4, 8))
        result = rotary_embed(x)
        assert result.extra["output"].shape == (2, 4, 8)

    def test_position_zero_unchanged(self):
        x = np.zeros((4, 8))
        result = rotary_embed(x)
        assert np.allclose(result.extra["output"], 0.0)

    def test_alias(self):
        assert rpemb is rotary_embed
