"""Tests for morie.fn.rcchy -- Cauchy random variates."""

import numpy as np
import pytest
from morie.fn.rcchy import rcchy


class TestRcchy:
    def test_length(self):
        """Output has correct length."""
        result = rcchy(100, seed=42)
        assert len(result) == 100

    def test_reproducible(self):
        """Same seed gives same output."""
        a = rcchy(10, seed=42)
        b = rcchy(10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_median_near_loc(self):
        """Sample median should be near loc for large n."""
        x = rcchy(10000, loc=5.0, seed=42)
        assert abs(np.median(x) - 5.0) < 0.5

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            rcchy(10, scale=0)

    def test_raises_bad_n(self):
        with pytest.raises(ValueError):
            rcchy(0)
