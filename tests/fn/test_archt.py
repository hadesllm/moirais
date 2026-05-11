"""Tests for morie.fn.archt -- ARCH(p)."""
import numpy as np
import pytest
from morie.fn.archt import arch_fit


class TestARCH:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(200) * 0.01
        res = arch_fit(r, p=1)
        assert res.extra["alpha"][0] >= 0

    def test_short_raises(self):
        with pytest.raises(ValueError):
            arch_fit(np.ones(5), p=1)

    def test_cheatsheet(self):
        from morie.fn.archt import cheatsheet
        assert isinstance(cheatsheet(), str)
