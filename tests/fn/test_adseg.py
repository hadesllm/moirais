"""Test adaptive_segment (adseg)."""
import numpy as np
from morie.fn.adseg import adaptive_segment, adseg
from morie.fn._containers import DescriptiveResult


class TestAdaptiveSegment:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) * 5])
        result = adaptive_segment(x, min_len=20)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "adaptive_segment"

    def test_segments_found(self):
        x = np.concatenate([np.ones(60), np.ones(60) * 10])
        result = adaptive_segment(x, min_len=20, threshold=2.0)
        assert result.value >= 1

    def test_boundaries(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        result = adaptive_segment(x, min_len=50)
        bounds = result.extra["boundaries"]
        assert bounds[0] == 0
        assert bounds[-1] == len(x)

    def test_alias(self):
        assert adseg is adaptive_segment
