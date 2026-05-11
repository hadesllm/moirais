"""Tests for morie.fn.varm -- VAR model."""
import numpy as np
import pytest
from morie.fn.varm import var_fit


class TestVAR:
    def test_basic(self):
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((50, 2))
        res = var_fit(Y, lags=1)
        assert res.name == "var_fit"
        assert res.extra["k"] == 2

    def test_short_raises(self):
        with pytest.raises(ValueError):
            var_fit(np.ones((2, 2)), lags=1)

    def test_cheatsheet(self):
        from morie.fn.varm import cheatsheet
        assert isinstance(cheatsheet(), str)
