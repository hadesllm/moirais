"""Tests for morie.fn.armam -- ARMA(p,q) model."""
import numpy as np
import pytest
from morie.fn.armam import arma_fit


class TestARMAFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(200)
        res = arma_fit(y, p=1, q=1)
        assert res.name == "arma_fit"
        assert "phi" in res.extra
        assert "theta" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            arma_fit(np.ones(5), p=1, q=1)

    def test_cheatsheet(self):
        from morie.fn.armam import cheatsheet
        assert isinstance(cheatsheet(), str)
