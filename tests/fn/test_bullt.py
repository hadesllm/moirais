"""Tests for moirais.fn.bullt -- time stretching."""

import numpy as np
from moirais.fn.bullt import time_stretch, bullt
from moirais.fn._containers import DescriptiveResult


class TestBullt:
    def test_alias(self):
        assert bullt is time_stretch

    def test_stretch(self):
        t = np.linspace(0, 1, 1024)
        sig = np.sin(2 * np.pi * 10 * t)
        result = time_stretch(sig, factor=2.0, window=256)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["output_length"] > len(sig)

    def test_compress(self):
        t = np.linspace(0, 1, 1024)
        sig = np.sin(2 * np.pi * 5 * t)
        result = time_stretch(sig, factor=0.5, window=256)
        assert result.extra["output_length"] < len(sig)
