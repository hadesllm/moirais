"""Tests for moirais.fn.wndwm -- Window functions."""

from moirais.fn.wndwm import window_function, wndwm
from moirais.fn._containers import DescriptiveResult


class TestWndwm:
    def test_alias(self):
        assert wndwm is window_function

    def test_hann(self):
        result = window_function(64, kind="hann")
        assert isinstance(result, DescriptiveResult)
        w = result.value["window"]
        assert len(w) == 64
        assert result.extra["coherent_gain"] < 1.0

    def test_rectangular(self):
        result = window_function(32, kind="rectangular")
        assert abs(result.extra["coherent_gain"] - 1.0) < 1e-10
