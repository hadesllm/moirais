"""Test blackman_window (wnblk)."""
import numpy as np
from moirais.fn.wnblk import blackman_window, wnblk
from moirais.fn._containers import DescriptiveResult


class TestWnblk:
    def test_basic(self):
        result = blackman_window(16)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "blackman_window"

    def test_length(self):
        result = blackman_window(64)
        assert len(result.extra["window"]) == 64

    def test_alias(self):
        assert wnblk is blackman_window
