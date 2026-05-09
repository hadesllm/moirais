"""Tests for moirais.fn.mavar -- Moving average smoother."""
import numpy as np
import pytest
from moirais.fn.mavar import moving_average


class TestMovingAverage:
    def test_simple(self):
        y = np.arange(20, dtype=float)
        res = moving_average(y, window=5, method="simple")
        assert len(res.extra["smoothed"]) == 20

    def test_exponential(self):
        y = np.arange(20, dtype=float)
        res = moving_average(y, window=5, method="exponential")
        assert res.extra["method"] == "exponential"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            moving_average(np.ones(10), method="bad")

    def test_cheatsheet(self):
        from moirais.fn.mavar import cheatsheet
        assert isinstance(cheatsheet(), str)
