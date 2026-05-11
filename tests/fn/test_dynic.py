"""Tests for morie.fn.dynic -- Dynamic linear model."""
import numpy as np
import pytest
from morie.fn.dynic import dlm_fit


class TestDLM:
    def test_local_level(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(50)) * 0.1 + rng.normal(0, 1, 50)
        res = dlm_fit(y, model="local_level")
        assert len(res.extra["level"]) == 50

    def test_local_trend(self):
        rng = np.random.default_rng(42)
        y = np.arange(30, dtype=float) + rng.normal(0, 1, 30)
        res = dlm_fit(y, model="local_trend")
        assert "trend" in res.extra

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            dlm_fit(np.ones(10), model="bad")

    def test_cheatsheet(self):
        from morie.fn.dynic import cheatsheet
        assert isinstance(cheatsheet(), str)
