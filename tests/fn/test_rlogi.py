"""Tests for moirais.fn.rlogi -- Logistic random variates."""

import numpy as np
import pytest
from moirais.fn.rlogi import rlogi


class TestRlogi:
    def test_length(self):
        """Output has correct length."""
        result = rlogi(100, seed=42)
        assert len(result) == 100

    def test_reproducible(self):
        """Same seed gives same output."""
        a = rlogi(10, seed=42)
        b = rlogi(10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_mean_near_loc(self):
        """Sample median should be near loc for large n."""
        x = rlogi(10000, loc=3.0, seed=42)
        assert abs(np.median(x) - 3.0) < 0.2

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            rlogi(10, scale=0)

    def test_raises_bad_n(self):
        with pytest.raises(ValueError):
            rlogi(0)
