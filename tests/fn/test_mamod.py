"""Tests for morie.fn.mamod -- MA(q) model."""
import numpy as np
import pytest
from morie.fn.mamod import ma_fit


class TestMAFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        res = ma_fit(y, q=1)
        assert res.name == "ma_fit"
        assert len(res.extra["theta"]) == 1

    def test_short_raises(self):
        with pytest.raises(ValueError):
            ma_fit(np.ones(3), q=1)

    def test_cheatsheet(self):
        from morie.fn.mamod import cheatsheet
        assert isinstance(cheatsheet(), str)
