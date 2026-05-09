"""Tests for moirais.fn.ma_ — moving average smoother."""

import numpy as np
import pytest

from moirais.fn.ma_ import ma_


class TestMa:
    """Tests for ma_()."""

    def test_centered_window(self):
        """Centered MA of constant series returns the constant."""
        series = np.full(20, 5.0)
        result = ma_(series, window=5, center=True)
        # Middle values should be 5.0
        assert result[5] == pytest.approx(5.0)

    def test_trailing_window(self):
        """Non-centered MA starts at index window-1."""
        series = np.arange(10, dtype=float)
        result = ma_(series, window=3, center=False)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)  # mean(0, 1, 2)

    def test_window_1_returns_copy(self):
        """Window=1 returns a copy of the original."""
        series = np.array([1.0, 2.0, 3.0])
        result = ma_(series, window=1)
        np.testing.assert_array_equal(result, series)

    def test_raises_empty(self):
        """Empty series raises ValueError."""
        with pytest.raises(ValueError):
            ma_(np.array([]))
