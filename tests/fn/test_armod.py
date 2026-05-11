"""Tests for morie.fn.armod -- AR(p) model."""
import numpy as np
import pytest
from morie.fn.armod import ar_fit


class TestARFit:
    def test_basic_ar1(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100)) * 0.1
        res = ar_fit(y, p=1)
        assert res.name == "ar_fit"
        assert len(res.extra["phi"]) == 1
        assert res.extra["n"] == 100

    def test_ar2(self):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(200)
        res = ar_fit(y, p=2)
        assert len(res.extra["phi"]) == 2

    def test_short_raises(self):
        with pytest.raises(ValueError):
            ar_fit(np.ones(2), p=1)

    def test_cheatsheet(self):
        from morie.fn.armod import cheatsheet
        assert isinstance(cheatsheet(), str)
