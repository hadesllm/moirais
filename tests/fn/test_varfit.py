"""Tests for moirais.fn.varfit — vector autoregression."""
import numpy as np
import pytest
from moirais.fn.varfit import var_fit


class TestVarFit:
    def test_bivariate(self):
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((200, 2))
        res = var_fit(Y, p=1)
        beta = res.extra["coefficients"]
        assert beta.shape == (3, 2)
        assert np.isfinite(res.extra["aic"])

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="Need T"):
            var_fit(np.ones((3, 2)), p=2)

    def test_univariate(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        res = var_fit(y, p=1)
        assert res.extra["k"] == 1
