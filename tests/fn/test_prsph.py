"""Tests for morie.fn.prsph -- SSIM."""

import numpy as np
from morie.fn.prsph import ssim, prsph
from morie.fn._containers import DescriptiveResult


class TestPrsph:
    def test_alias(self):
        assert prsph is ssim

    def test_identical(self):
        x = np.random.default_rng(42).random(100)
        result = ssim(x, x, win_size=7)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.99

    def test_different(self):
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)
        result = ssim(x, y)
        assert result.value < 0.5
