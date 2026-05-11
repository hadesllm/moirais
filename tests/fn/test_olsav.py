"""Test overlap_save (olsav)."""
import numpy as np
from morie.fn.olsav import overlap_save, olsav
from morie.fn._containers import DescriptiveResult


class TestOlsav:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(128)
        h = np.array([0.25, 0.5, 0.25])
        result = overlap_save(x, h, block_size=32)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "overlap_save"

    def test_matches_direct(self):
        x = np.random.default_rng(42).standard_normal(128)
        h = np.array([0.25, 0.5, 0.25])
        result = overlap_save(x, h, block_size=32)
        direct = np.convolve(x, h)
        n = min(len(result.extra["output"]), len(direct))
        assert np.allclose(result.extra["output"][:n], direct[:n], atol=1e-10)

    def test_alias(self):
        assert olsav is overlap_save
