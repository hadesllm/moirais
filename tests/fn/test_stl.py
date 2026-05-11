"""Tests for morie.fn.stl — STL decomposition."""
import numpy as np
import pytest
from morie.fn.stl import stl_decompose


class TestSTL:
    def test_basic(self):
        t = np.arange(48)
        y = np.sin(2 * np.pi * t / 12) + 0.01 * t + np.random.default_rng(42).standard_normal(48) * 0.1
        res = stl_decompose(y, period=12)
        assert len(res.extra["trend"]) == 48

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            stl_decompose(np.ones(10), period=12)
