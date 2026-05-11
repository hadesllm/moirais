"""Test pca_overlap (pcaov)."""
import numpy as np
from morie.fn.pcaov import pca_overlap, pcaov
from morie.fn._containers import DescriptiveResult


class TestPcaov:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        result = pca_overlap(x, window=64, hop=32, n_components=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "pca_overlap"
        assert 0.0 <= result.value <= 1.0

    def test_n_frames(self):
        x = np.random.default_rng(0).standard_normal(256)
        r = pca_overlap(x, window=64, hop=32)
        expected = (256 - 64) // 32 + 1
        assert r.extra["n_frames"] == expected

    def test_alias(self):
        assert pcaov is pca_overlap
