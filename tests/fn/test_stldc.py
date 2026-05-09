"""Tests for moirais.fn.stldc -- STL decomposition."""
import numpy as np
import pytest
from moirais.fn.stldc import stl_decompose


class TestSTL:
    def test_basic(self):
        rng = np.random.default_rng(42)
        t = np.arange(48)
        y = 10 + 0.5 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 48)
        res = stl_decompose(y, period=12)
        assert res.extra["trend"] is not None
        assert len(res.extra["seasonal"]) == 48

    def test_short_raises(self):
        with pytest.raises(ValueError):
            stl_decompose(np.ones(10), period=12)

    def test_cheatsheet(self):
        from moirais.fn.stldc import cheatsheet
        assert isinstance(cheatsheet(), str)
