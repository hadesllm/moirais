"""Tests for moirais.fn.hurst — Hurst exponent."""

import numpy as np
import pytest

from moirais.fn.hurst import hurst


class TestHurst:
    """Tests for hurst()."""

    def test_random_walk_persistent(self):
        """Random walk should give H > 0.5 (persistent)."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(500))
        result = hurst(y)
        assert 0 <= result["H"] <= 1
        assert result["H"] > 0.5

    def test_white_noise_near_half(self):
        """White noise should give H near 0.5."""
        rng = np.random.default_rng(7)
        y = rng.standard_normal(1000)
        result = hurst(y)
        assert 0.3 <= result["H"] <= 0.7

    def test_interpretation_string(self):
        """Interpretation is one of the known values."""
        rng = np.random.default_rng(99)
        y = rng.standard_normal(100)
        result = hurst(y)
        assert result["interpretation"] in ("persistent", "anti-persistent", "random")

    def test_raises_short(self):
        """Short series raises ValueError."""
        with pytest.raises(ValueError, match="20"):
            hurst(np.array([1, 2, 3]))
